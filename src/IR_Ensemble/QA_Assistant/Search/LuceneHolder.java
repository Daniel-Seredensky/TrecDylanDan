package src.IR_Ensemble.QA_Assistant.Search;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Lazily initialised, JVM‑wide Lucene resources.
 * Thread‑safe courtesy of class‑initialisation + 'synchronized'.
 */
public final class LuceneHolder {

    private static final Path INDEX_PATH = Paths.get("/Volumes/X9 Pro/MarcoIndex");

    private static FSDirectory   DIR;
    private static DirectoryReader READER;
    private static IndexSearcher   SEARCHER;

    // handles *outer* SearcherDaemon requests
    private static final int REQUEST_THREADS =
        Math.max(3, Runtime.getRuntime().availableProcessors() / 2);
    private static final ExecutorService REQUEST_POOL =
        Executors.newFixedThreadPool(REQUEST_THREADS);

    // heavy lifting for Lucene work
    private static final int LUCENE_THREADS =
        Math.max(10, Runtime.getRuntime().availableProcessors() * 2);
    private static final ExecutorService LUCENE_POOL =
        Executors.newFixedThreadPool(LUCENE_THREADS);

    public static ExecutorService getRequestPool() { return REQUEST_POOL; }
    public static ExecutorService getLucenePool()  { return LUCENE_POOL; }
    private LuceneHolder() {}    // no instances

    /** Initialise once, cheap thereafter */
    private synchronized static void init() throws IOException {
        if (READER != null) return;                    // already done

        DIR      = FSDirectory.open(INDEX_PATH);
        READER   = DirectoryReader.open(DIR);
        SEARCHER = new IndexSearcher(READER);
        SEARCHER.setSimilarity(new BM25Similarity());
    }

    public static IndexSearcher getSearcher() throws IOException {
        init();
        return SEARCHER;
    }

    public static DirectoryReader getReader() throws IOException {
        init();
        return READER;
    }

    static {
        /* one JVM‑wide hook – installs exactly once */
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                if (READER != null)  READER.close();      // closes sub‑readers & fd’s
                if (DIR != null)     DIR.close();
            } catch (Exception e) {
                System.err.println("Lucene close failed: " + e);
            }
            REQUEST_POOL.shutdown();
            LUCENE_POOL.shutdown();                              // stop accepting new work
            try {                                         // give tasks 5 s to finish
                if (!REQUEST_POOL.awaitTermination(5, TimeUnit.SECONDS))
                    REQUEST_POOL.shutdownNow();
                if (!LUCENE_POOL.awaitTermination(5, TimeUnit.SECONDS))
                    LUCENE_POOL.shutdownNow();
            } catch (InterruptedException ie) {
                LUCENE_POOL.shutdownNow();
                REQUEST_POOL.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }, "lucene-shutdown"));
    }
}
