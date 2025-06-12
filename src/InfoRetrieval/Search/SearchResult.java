package src.InfoRetrieval.Search;

import java.util.Map;
import java.util.HashMap;

public class SearchResult {
    private final String url;
    private final String id;
    private final String content;
    private final String highlightedContent;
    private double rrfScore;
    private final Map<String, Integer> queryRanks = new HashMap<>();
    private final Map<String, Float> queryScores = new HashMap<>();
    private final String segmentId;
    private final String title, headings;

    public SearchResult(String url, String id, String segmentId, String content,String title, String headings, String highlightedContent,
                        String queryId, int rank, float score) {
        this.url = url;
        this.id = id;
        this.segmentId = segmentId;
        this.content = content;
        this.highlightedContent = highlightedContent;
        this.queryRanks.put(queryId, rank);
        this.queryScores.put(queryId, score);
        this.title = title;
        this.headings = headings;
    }

    public void addQueryResult(String qid, int rank, float score) {
        queryRanks.put(qid, rank);
        queryScores.put(qid, score);
    }

    public void calculateRrfScore(double k) {
        this.rrfScore = queryRanks.values().stream()
                              .mapToDouble(r -> 1.0 / (k + r)).sum();
    }

    public int getQueryCount() { return queryRanks.size(); }
    public String getUrl() { return url; }
    public String getId() { return id; }
    public String getSegmentId() { return segmentId; }
    public String getHighlightedContent() { return highlightedContent; }
    public String getContent (){ return content;}
    public double getRrfScore() { return rrfScore; }
    public String getTitle() {return title;}
    public String getHeadings() {return headings;}
    public Map<String, Integer> getQueryRanks() { return queryRanks; }
    public Map<String, Float> getQueryScores() { return queryScores; }
}