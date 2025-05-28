package InfoRetrieval;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexOptions;
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

        String jsonlPath = "clueweb/TREC-LR-2024/T2/trec-2024-lateral-reading-task2-baseline-documents.jsonl";
        String indexDir = "DerivedData/CluewebIndex";

        // Count total lines to calculate shards
        long totalLines = countLines(jsonlPath);
        long linesPerShard = (totalLines + NUM_SHARDS - 1) / NUM_SHARDS;

        // Setup Lucene index writer
        Directory dir = NIOFSDirectory.open(Paths.get(indexDir));
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());

        // Performance tuning
        config.setRAMBufferSizeMB(RAM_BUFFER_MB);
        config.setMergeScheduler(new ConcurrentMergeScheduler());
        TieredMergePolicy mergePolicy = (TieredMergePolicy) config.getMergePolicy();
        mergePolicy.setMaxMergeAtOnce(4);
        mergePolicy.setSegmentsPerTier(4);
        // Use BM25 similarity (default) or customize
        config.setSimilarity(new BM25Similarity());

        IndexWriter writer = new IndexWriter(dir, config);

        // Prepare custom field for content
        FieldType bm25Field = new FieldType();
        bm25Field.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        bm25Field.setOmitNorms(false);
        bm25Field.setStoreTermVectors(false);
        bm25Field.setStored(true);  // no storage needed CHANGE THIS IF YOU WANT TEXT RETRIEVAL
        bm25Field.freeze();

        // Thread pool for shards
        ExecutorService executor = Executors.newFixedThreadPool(NUM_SHARDS);
        for (int shard = 0; shard < NUM_SHARDS; shard++) {
            final int shardNum = shard;
            executor.submit(() -> {
                try {
                    indexShard(jsonlPath, writer, bm25Field, shardNum, linesPerShard);
                } catch (Exception e) {
                    System.err.println("Error in shard " + shardNum);
                    e.printStackTrace();
                }
            });
        }

        // Shutdown
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);
        writer.close();
    }

    private static long countLines(String filePath) throws IOException {
        long count = 0;
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(filePath))) {
            while (reader.readLine() != null) {
                count++;
            }
        }
        return count;
    }

    private static void indexShard(String jsonlPath,
                                   IndexWriter writer,
                                   FieldType bm25Field,
                                   int shardNum,
                                   long linesPerShard) throws IOException {
        long start = shardNum * linesPerShard;
        long end = start + linesPerShard;
        ObjectMapper mapper = new ObjectMapper();

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(jsonlPath))) {
            String line;
            long lineNum = 0;
            while ((line = reader.readLine()) != null) {
                if (lineNum >= start && lineNum < end) {
                    JsonNode node = mapper.readTree(line);
                    String url = node.path("URL").asText();
                    String id = node.path("ClueWeb22-ID").asText();
                    String content = node.path("Clean-Text").asText();

                    Document doc = new Document();
                    doc.add(new StringField("url", url, Field.Store.YES));
                    doc.add(new StringField("id", id, Field.Store.YES));
                    doc.add(new Field("content", content, bm25Field));
                    writer.addDocument(doc);
                }
                if (lineNum >= end) break;
                lineNum++;
            }
        }
        System.out.println("Shard " + shardNum + " completed: lines " + start + " to " + (end - 1));
    }
}
