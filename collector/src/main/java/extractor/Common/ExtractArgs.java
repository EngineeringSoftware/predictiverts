package extractor.Common;

public class ExtractArgs {

    public int MaxCodeLength = -1;
    public int MinCodeLength = 1;

    public int MaxPathLength;
    public int MaxPathWidth;
    public int MaxChildId = 3;

    public ExtractArgs( int MaxPathLength, int MaxPathWidth) {
        this.MaxPathLength = MaxPathLength;
        this.MaxPathWidth = MaxPathWidth;
    }

}
