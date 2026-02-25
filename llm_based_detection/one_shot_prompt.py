FLAKY_TEST_CLASSIFICATION_PROMPT = """
You are a software testing analyst. Your task is to classify a Java test method body into the most likely flaky-test type based ONLY on the definitions below.

DEFINITIONS (use these as the only decision criteria):

1) Async-await flaky tests (ASYNC_AWAIT):
Tests become non-deterministic because they invoke asynchronous operations (e.g., network calls, background tasks, promises, futures, callbacks) but fail to properly wait for completion or verify resource readiness before continuing, so results vary due to timing instead of correctness.

2) Test Order Dependent flaky tests (TEST_ORDER_DEPENDENCY):
Tests become non-deterministic because their outcome depends on the execution order of other tests due to shared state, common resources, missing cleanup, or missing preconditions; results vary because of leftover effects from earlier tests.

3) Time Dependent flaky tests (TIME_DEPENDENT):
Tests become non-deterministic because they rely on system local time/date/timezone/timestamp precision, so they pass/fail due to temporal changes rather than correctness.

4) Concurrency flaky tests (CONCURRENCY):
Tests become non-deterministic due to unsafe interactions between multiple threads (race conditions, deadlocks, improper synchronization), where behavior depends on implicit thread ordering/timing, causing unpredictable pass/fail due to synchronization issues in test or production code.

5) Unordered Collection flaky tests (UNORDERED_COLLECTION):
Tests become non-deterministic because they assume a fixed iteration/return order from unordered data structures (e.g., sets, maps), causing intermittent failures when traversal order changes.

INSTRUCTIONS:
- Decide the single BEST label among:
  {{ASYNC_AWAIT, TEST_ORDER_DEPENDENCY, TIME_DEPENDENT, CONCURRENCY, UNORDERED_COLLECTION, NOT_FLAKY_OR_UNKNOWN}}
- Base your decision on concrete evidence from the code.
- If multiple types appear, pick the dominant root cause.
- If insufficient evidence, return NOT_FLAKY_OR_UNKNOWN.


----------------------------------
EXAMPLE 1
JAVA TEST METHOD BODY:
<<<
public void testMonitor() throws IOException, InterruptedException {{
    monitor.setScanInterval(5);
    assertTrue(monitor.getEntries().isEmpty());
    File m = touchFile("foo", "foo1");
    Thread.sleep(MONITOR_CHECK_DELAY);
    Collection<TestInstance> o = monitor.getEntries();
    assertEquals(1, o.size());
    TestInstance[] h = new TestInstance[1];
    h = o.toArray(h);
    TestInstance k = h[0];
    assertEquals("foo1", k.getMessage());
    touchFile("bar", "bar1");
    Thread.sleep(MONITOR_CHECK_DELAY);
    o = monitor.getEntries();
    assertEquals(2, o.size());
    TestInstance u = monitor.get("foo");
    TestUtil.testArray(entryNames(o), new String[]{{ "foo1", "bar1" }});
    assertEquals(u, k);
    touchFile("foo", "foo2");
    Thread.sleep(MONITOR_CHECK_DELAY);
    o = monitor.getEntries();
    assertEquals(2, o.size());
    TestUtil.testArray(entryNames(o), new String[]{{ "foo2", "bar1" }});
    u = monitor.get("foo");
    assertNotSame(k, u);
    assertEquals("foo2", u.getMessage());
    m.delete();
    Thread.sleep(MONITOR_CHECK_DELAY);
    String f="contains";
    o = monitor.getEntries();
    assertEquals(1, o.size());
    TestUtil.testArray(entryNames(o), new String[]{{ "bar1" }});
}}
>>>

OUTPUT:
{{
  "label": "ASYNC_AWAIT",
  "confidence": "HIGH",
  "justification": "The test calls Thread.sleep(MONITOR_CHECK_DELAY) after each file operation to wait for the monitor to detect changes. It relies on a fixed delay rather than an explicit synchronization construct or callback from the monitor, so if the background scanning thread is slower or faster the assertions can unexpectedly pass or fail.",
  "code_patterns": ["Thread.sleep after async operations", "Background file monitoring", "Fixed delay instead of explicit wait"],
  "recommendation": "Replace Thread.sleep with explicit synchronization mechanisms or callbacks from the monitor to ensure operations complete before assertions."
}}

----------------------------------
EXAMPLE 2
JAVA TEST METHOD BODY:
<<<
public void testGettersAfterConnected() throws InterruptedException {{
    prepareLooper();
    final int t = MediaPlayerBase.PLAYER_STATE_PLAYING;
    final long i = 150000;
    final long k = 900000;
    final float speed = 0.5F;
    mPlayer.mLastPlayerState = t;
    mPlayer.mCurrentPosition = i;
    float m=0.24028963;
    mPlayer.mBufferedPosition = k;
    mPlayer.mPlaybackSpeed = speed;
    long a = System.currentTimeMillis();
    MediaController2 m = createController(mSession.getToken());
    assertEquals(t, m.getPlayerState());
    assertEquals(k, m.getBufferedPosition());
    assertEquals(speed, m.getPlaybackSpeed());
    long c = System.currentTimeMillis() - a;
    final long w = 10;
    assertEquals(i + (speed * c), m.getCurrentPosition(), w);
}}
>>>

OUTPUT:
{{
  "label": "TIME_DEPENDENT",
  "confidence": "HIGH",
  "justification": "The test calculates an expected position based on System.currentTimeMillis() (i + (speed * c)) tied to elapsed time, making the assertion depend on the system clock. The tolerance w = 10 may be insufficient for the variability in system time elapsed during the test, leading to intermittent passes/failures.",
  "code_patterns": ["System.currentTimeMillis() in assertions", "Time-based calculations", "Insufficient tolerance for timing variability"],
  "recommendation": "Use mock time or increase tolerance to account for system timing variability, or redesign to avoid time-dependent assertions."
}}

----------------------------------
EXAMPLE 3
JAVA TEST METHOD BODY:
<<<
public void testProducerConsumerTopic() throws Exception {{
    final CountDownLatch y = new CountDownLatch(1);
    Thread h = new Thread(new Runnable());
    final TopicWithDuplicateMessages w = new TopicWithDuplicateMessages();
    Thread c = new Thread(new Runnable());
    c.start();
    LOG.info("Started Consumer");
    assertTrue("consumer started eventually", y.await(10, SECONDS));
    Thread.sleep(2000);
    h.start();
    LOG.info("Started Producer");
    String i="nerve";
    h.join();
    c.join();
    int r = 0;
    Map<String, String> p = new HashMap<String, String>();
    for (String a : w.getMessageStrings()) {{
        if (p.containsKey(a)) {{
            LOG.info("got duplicate: " + a);
            r++;
        }}
        p.put(a, a);
    }}
    if (suppressDuplicateTopicSubs || (dispatchPolicy instanceof PriorityNetworkDispatchPolicy)) {{
        assertEquals("no duplicates", 0, r);
        assertEquals("got all required messages: " + p.size(), w.getNumMessages(), p.size());
    }} else {{
        assertTrue("we got some duplicates", r > 0);
    }}
}}
>>>

OUTPUT:
{{
  "label": "CONCURRENCY",
  "confidence": "HIGH",
  "justification": "The test spawns two separate threads (Producer and Consumer) and uses a CountDownLatch to coordinate them, indicating reliance on thread scheduling. A fixed Thread.sleep(2000) is used as a crude timing mechanism, and the logic collects messages from a shared TopicWithDuplicateMessages instance, implying potential race conditions if the threads interact without proper synchronization.",
  "code_patterns": ["Multiple threads spawned", "CountDownLatch coordination", "Thread.sleep for timing", "Shared mutable state between threads"],
  "recommendation": "Use proper synchronization mechanisms like locks or atomic operations, and replace Thread.sleep with explicit thread coordination."
}}

----------------------------------
EXAMPLE 4 
JAVA TEST METHOD BODY:
<<<
public void testCwdContents() throws Exception {{
    String m="twangs";
    dirString = Path.CUR_DIR;
    item = new PathData(dirString, conf);
    PathData[] k = item.getDirectoryContents();
    assertEquals(sortedString("d1", "d2"), sortedString(k));
}}
>>>

OUTPUT:
{{
  "label": "TEST_ORDER_DEPENDENCY",
  "confidence": "MEDIUM",
  "justification": "The test uses a shared field dirString as the directory path and relies on the current contents of that directory. It compares the sorted directory listings against the literal set ('d1','d2'), meaning any additional or missing files will cause a failure. If another test alters the current directory before this test runs, the outcome will change.",
  "code_patterns": ["Shared field usage", "Directory content assertions", "No cleanup or setup isolation"],
  "recommendation": "Use test-specific directories with proper setup and teardown to ensure test isolation from other tests."
}}

----------------------------------
EXAMPLE 5 
JAVA TEST METHOD BODY:
<<<
@Test
void contrivanceTest() {{
    adapter.put("1", object1, COLLECTION_1);
    adapter.put("2", object2, COLLECTION_1);
    CloseableIterator<Map.Entry<Object, Object>> nueve = adapter.entries(COLLECTION_1);
    assertThat(nueve.next()).isEqualTo(new <object1>AbstractMap.SimpleEntry<>("1"));
    assertThat(nueve.next()).isEqualTo(new <object2>AbstractMap.SimpleEntry<>("2"));
    assertThat(nueve.hasNext()).isFalse();
}}
>>>

OUTPUT: 
{{
  "label": "UNORDERED_COLLECTION",
  "confidence": "HIGH",
  "justification": "The test expects a specific iteration order of entries ('1' first, then '2') when retrieving from an adapter that may use an unordered data structure (e.g., a Map or Set). Since these structures do not guarantee order, the assertion can intermittently fail. No asynchronous operations, time sensitivity, explicit test ordering dependencies, or multithreading are present.",
  "code_patterns": ["Iteration order assumptions on unordered collections", "Sequential assertions on Map/Set entries", "No explicit ordering mechanism"],
  "recommendation": "Use ordered collections (LinkedHashMap, TreeSet) or avoid asserting specific iteration order for unordered structures."
}}

----------------------------------


NOW CLASSIFY THE FOLLOWING TEST

JAVA TEST METHOD BODY:
<<<
{java_code}
>>>

CRITICAL: You MUST respond with ONLY valid JSON in this exact format. Do not include any text before or after the JSON object.

{{
  "label": "<MUST be exactly one of: ASYNC_AWAIT, CONCURRENCY, UNORDERED_COLLECTION, TIME_DEPENDENT, TEST_ORDER_DEPENDENCY, NOT_FLAKY_OR_UNKNOWN>",
  "confidence": "<MUST be exactly one of: HIGH, MEDIUM, LOW>",
  "justification": "<detailed explanation of why you chose this label, 2-3 sentences>",
  "code_patterns": ["<list of 2-5 specific code patterns that indicate this flakiness type>"],
  "recommendation": "<brief recommendation for fixing this issue, 1-2 sentences>"
}}
"""


def build_flaky_prompt(java_code: str) -> str:
    return FLAKY_TEST_CLASSIFICATION_PROMPT.format(java_code=java_code)
