package src.InfoRetrieval.Search;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.index.ConcurrentMergeScheduler;

public class JsonIndexer {
    private static final int NUM_SHARDS = 4;
    private static final int RAM_BUFFER_MB = 64;

    public static void main(String[] args) throws Exception {
        // <-- point this at your MS MARCO v2.1 JSONL
        String jsonlPath = "SampleMarco/FixedSampleData.jsonl";
        String indexDir  = "DerivedData/MSMarcoIndex";

        long totalLines    = countLines(jsonlPath);
        long linesPerShard = (totalLines + NUM_SHARDS - 1) / NUM_SHARDS;

        Directory dir = NIOFSDirectory.open(Paths.get(indexDir));
        IndexWriterConfig config = new IndexWriterConfig(new EnglishAnalyzer());
        config.setRAMBufferSizeMB(RAM_BUFFER_MB);
        config.setMergeScheduler(new ConcurrentMergeScheduler());
        TieredMergePolicy mergePolicy = (TieredMergePolicy) config.getMergePolicy();
        mergePolicy.setMaxMergeAtOnce(4);
        mergePolicy.setSegmentsPerTier(4);
        config.setSimilarity(new BM25Similarity());

        try (IndexWriter writer = new IndexWriter(dir, config)) {
            ExecutorService executor = Executors.newFixedThreadPool(NUM_SHARDS);
            for (int shard = 0; shard < NUM_SHARDS; shard++) {
                final int shardNum = shard;
                executor.submit(() -> {
                    try {
                        indexShard(jsonlPath, writer, shardNum, linesPerShard);
                    } catch (Exception e) {
                        System.err.println("Error in shard " + shardNum);
                        e.printStackTrace();
                    }
                });
            }
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.HOURS);
        }
    }

    private static long countLines(String filePath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(filePath))) {
            return reader.lines().count();
        }
    }

    private static void indexShard(String jsonlPath,
                                   IndexWriter writer,
                                   int shardNum,
                                   long linesPerShard) throws IOException {
        long start = shardNum * linesPerShard;
        long end   = start + linesPerShard;
        ObjectMapper mapper = new ObjectMapper();

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(jsonlPath))) {
            String line;
            long   lineNum = 0;

            while ((line = reader.readLine()) != null) {
                if (lineNum >= start && lineNum < end) {
                    JsonNode node = mapper.readTree(line);

                    // pull out each field from the JSON
                    String segmentId = node.path("segment_id").asText();
                    String url       = node.path("url").asText();
                    String title     = node.path("title").asText();
                    String headings  = node.path("headings").asText();
                    String segment   = node.path("segment").asText();

                    Document doc = new Document();
                    // untokenized fields
                    doc.add(new StringField("segment_id", segmentId, Store.YES));
                    doc.add(new StringField("url",        url,       Store.YES));
                    // tokenized fields
                    doc.add(new TextField("title",    title,    Store.YES));
                    doc.add(new TextField("headings", headings, Store.YES));
                    doc.add(new TextField("segment",  segment,  Store.YES));

                    writer.addDocument(doc);
                }
                if (lineNum >= end) {
                    break;
                }
                lineNum++;
            }
        }
        System.out.println("Shard " + shardNum + " completed: lines " + start + " to " + (end - 1));
    }
}
