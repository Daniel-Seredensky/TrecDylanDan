// src/InfoRetrieval/Search/RankedDocument.java
package src.InfoRetrieval.Search;

public class RankedDocument {
    private final String url;
    private final String segmentId;
    private final String fullDocId;
    private final String content;
    private final String highlightedContent;
    private final float  bm25Score;
    private final int    rank;
    private final String title;
    private final String headings;

    public RankedDocument(String url,
                          String segmentId,
                          String content,
                          String title,
                          String headings,
                          String highlightedContent,
                          float bm25Score,
                          int   rank) {
        this.url                 = url;
        this.segmentId           = segmentId;
        this.fullDocId           = extractFullDocId(segmentId);
        this.content             = content;
        this.highlightedContent  = highlightedContent;
        this.bm25Score           = bm25Score;
        this.rank                = rank;
        this.title = title;
        this.headings = headings;
    }

    private static String extractFullDocId(String segmentId) {
        int idx = segmentId.indexOf('#');
        return (idx >= 0) ? segmentId.substring(0, idx) : segmentId;
    }

    public String getUrl()               { return url; }
    public String getSegmentId()         { return segmentId; }
    public String getFullDocId()         { return fullDocId; }
    public String getContent()           { return content; }
    public String getTitle() {return title; }
    public String getHeadings() {return headings;}
    public String getHighlightedContent(){ return highlightedContent; }
    public float  getBm25Score()         { return bm25Score; }
    public int    getRank()              { return rank; }
}
