package Indexer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * TextFileIndexer.java creates a Lucene index in the specified index directory for text files.
 * 
 * This version uses TF-IDF scoring (via ClassicSimilarity) and indexes the following fields:
 * <ul>
 *   <li>content: Full text (or, for Gutenberg files, only the text between
 *       *** START OF THE PROJECT GUTENBERG EBOOK and *** END OF THE PROJECT GUTENBERG EBOOK).</li>
 *   <li>stemcontent: The text after applying Porter stemming.</li>
 *   <li>stopcontent: The text after removing stop words.</li>
 *   <li>author: The author's name (extracted from a line containing "Author:").</li>
 *   <li>title: The document title (from a line containing "Title:"; if absent, the filename is used).</li>
 *   <li>filename: The name of the .txt file.</li>
 *   <li>filepath: The full path to the .txt file.</li>
 *   <li>modified: The file's last modified timestamp.</li>
 * </ul>
 * 
 * Usage:
 * <pre>
 *   TextFileIndexer.run(dataDirPath, indexDirPath, mode, isGutenberg);
 * </pre>
 * 
 * @version March 2025
 * @author A cs Professor
 * @author adapted by Daniel Seredensky, Oliwia, Ojo, William 
 */
public class TextFileIndexer {
    private static IndexingResult mostRecentIndexingResult;

    public static IndexingResult  getMostRecentIndexingResult() {
        return mostRecentIndexingResult == null ? new IndexingResult(-1, -1, -1) : mostRecentIndexingResult;
    }

    // Helper classes to store indexing results and document info from the index.
    public static class IndexingResult {
        public int added;
        public int changed;
        public int removed;
        public double elapsedTime;

        public IndexingResult(int added, int changed, int removed) {
            this.added = added;
            this.changed = changed;
            this.removed = removed;
        }

        public void addElapsedTime(double elapsedTime) {
            this.elapsedTime = elapsedTime;
        }
    }

    static class DocumentInfo {
        long modifiedTime;

        public DocumentInfo(long modifiedTime) {
            this.modifiedTime = modifiedTime;
        }
    }

    /**
     * Replaces the main method. Expects the following parameters:
     * @param dataDirPath  The directory containing .txt files.
     * @param indexDirPath The directory where the index will be stored.
     * @param option       Optional indexing mode: "new", "changed", or "missing".
     * @param isGutenberg  True if files are Gutenberg-formatted.
     */
    public static double run(String dataDirPath, String indexDirPath, String option, boolean isGutenberg, boolean verbose) {
        if (dataDirPath == null || indexDirPath == null) {
            System.err.println("Usage: run <dataDirPath> <indexDirPath> [new|changed|missing]");
            return -1;
        }
        
        // Validate optional indexing mode, if provided.
        if (option != null) {
            option = option.toLowerCase();
            if (!option.equals("new") && !option.equals("changed") && !option.equals("missing")) {
                System.err.println("Error: Invalid indexing option specified. Must be one of: new, changed, missing");
                return -1;
            }
        }
        
        try {
            long startTime = System.currentTimeMillis();
            IndexingResult result = indexTextFiles(dataDirPath, indexDirPath, option, isGutenberg);
            long elapsedTime = System.currentTimeMillis() - startTime;
            result.addElapsedTime(elapsedTime);
            mostRecentIndexingResult = result;

            if (verbose) {
                System.out.println("Indexing completed.");
                System.out.println("Documents added: " + result.added);
                System.out.println("Documents changed: " + result.changed);
                System.out.println("Documents removed: " + result.removed);
                System.out.println("Indexing time: " + elapsedTime + " ms");
            }
            return elapsedTime;
        } catch (IOException e) {
            e.printStackTrace();
            return -1;
        }
    }

    /**
     * indexTextFiles indexes files from the given data directory according to the provided option.
     *
     * @param dataDirPath  Path to the directory containing .txt files.
     * @param indexDirPath Path to the directory to write the index.
     * @param option       Optional index mode: "new", "changed", or "missing" (null means index all files).
     * @param isGutenberg  If true, only index content between the Gutenberg markers.
     * @return An IndexingResult with counts for added, changed, and removed documents.
     * @throws IOException
     */
    public static IndexingResult indexTextFiles(String dataDirPath, String indexDirPath, String option, boolean isGutenberg) throws IOException {
        Directory indexDir = FSDirectory.open(Paths.get(indexDirPath));
        
        // Use the shared analyzer from TextIndexingHelper
        StandardAnalyzer analyzer = TextIndexingHelper.THREAD_LOCAL_ANALYZER.get();
    
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        // Set TF-IDF scoring via ClassicSimilarity.
        config.setSimilarity(new ClassicSimilarity());
        IndexWriter writer = new IndexWriter(indexDir, config);
    
        int added = 0;
        int changed = 0;
        int removed = 0;
        
        // Define batch size based on document type
        // For Gutenberg files (large), commit after every 300 documents
        // For Cranfield files (small), commit at the end
        // This is because of a memory issue when indexing repeatedly as done in the Index proctor shell scrips
        final int BATCH_SIZE = isGutenberg ? 300 : 1400;
        int batchCount = 0;
    
        // Build a map of already indexed documents (by filepath)
        Map<String, DocumentInfo> indexDocs = new HashMap<>();
        if (DirectoryReader.indexExists(indexDir)) {
            DirectoryReader reader = DirectoryReader.open(indexDir);
            for (LeafReaderContext leaf : reader.leaves()) {
                LeafReader leafReader = leaf.reader();
                Bits liveDocs = leafReader.getLiveDocs();
                int maxDoc = leafReader.maxDoc();
                for (int i = 0; i < maxDoc; i++) {
                    if (liveDocs != null && !liveDocs.get(i)) continue;
                    Document doc = leafReader.document(i);
                    String filepath = doc.get("filepath");
                    String modifiedStr = doc.get("modified");
                    long modifiedTime = Long.parseLong(modifiedStr);
                    indexDocs.put(filepath, new DocumentInfo(modifiedTime));
                }
            }
            reader.close();
        }
    
        File dataDir = new File(dataDirPath);
        File[] files = dataDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".txt"));
    
        // Build a set of file paths present in the directory for "missing" processing 
        Set<String> currentFilePaths = new HashSet<>(files.length);
        if (files != null) {
            for (File file : files) {
                String filePath = file.getAbsolutePath(); 
                currentFilePaths.add(filePath);
                DocumentInfo info = indexDocs.get(filePath);
    
                boolean documentProcessed = false;
                
                if (option == null) {
                    // Default: index all files – add if new or update if modified.
                    if (info == null) {
                        writer.addDocument(TextIndexingHelper.createDocument(file, isGutenberg));
                        added++;
                        documentProcessed = true;
                    } else if (file.lastModified() > info.modifiedTime) {
                        writer.updateDocument(new Term("filepath", file.getAbsolutePath()), 
                                             TextIndexingHelper.createDocument(file, isGutenberg));
                        changed++;
                        documentProcessed = true;
                    }
                } else if (option.equals("new")) {
                    // Only add files not already in the index.
                    if (info == null) {
                        writer.addDocument(TextIndexingHelper.createDocument(file, isGutenberg));
                        added++;
                        documentProcessed = true;
                    }
                } else if (option.equals("changed")) {
                    // Only update files that are already indexed and have been modified
                    if (info != null && file.lastModified() > info.modifiedTime) {
                        writer.updateDocument(new Term("filepath", file.getAbsolutePath()), 
                                             TextIndexingHelper.createDocument(file, isGutenberg));
                        changed++;
                        documentProcessed = true;
                    }
                }
                
                // Commit in batches based on document type
                if (documentProcessed) {
                    batchCount++;
                    if (batchCount >= BATCH_SIZE) {
                        writer.commit();
                        batchCount = 0;
                    }
                }
            }
        }
    
        // Option "missing": remove indexed documents whose files no longer exist.
        if (option != null && option.equals("missing")) {
            int deleteBatchCount = 0;
            for (String filepath : indexDocs.keySet()) {
                if (!currentFilePaths.contains(filepath)) {
                    writer.deleteDocuments(new Term("filepath", filepath));
                    removed++;
                    
                    deleteBatchCount++;
                    if (deleteBatchCount >= BATCH_SIZE) {
                        writer.commit();
                        deleteBatchCount = 0;
                    }
                }
            }
        }
    
        // Final commit for any remaining documents
        writer.commit();
        writer.close();
        return new IndexingResult(added, changed, removed);
    }
    
    public static void main(String[] args) {
        String indexDir = "indexData";
        String dataDir = "data";
        boolean isGutenberg = true; 
        String mode = null;
        double time = TextFileIndexer.run(dataDir,indexDir,mode,isGutenberg,false);
        System.out.println(""+time + "\n");
        System.gc();
    }
}
