using System.Runtime.InteropServices;

namespace HypercubeGenerativeApi.Interop;

/// <summary>
/// P/Invoke declarations for hypercube_generative.dll
/// </summary>
public static class GenerativeInterop
{
    private const string DllName = "hypercube_generative.dll";
    private const CallingConvention CallConv = CallingConvention.Cdecl;

    // ==========================================================================
    // Cache Management
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void gen_vocab_clear();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void gen_bigram_clear();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void gen_attention_clear();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern int gen_vocab_count();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern int gen_bigram_count();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern int gen_attention_count();

    // ==========================================================================
    // Cache Loading
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern long gen_load_vocab();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern long gen_load_bigrams();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern long gen_load_attention();

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern IntPtr gen_load_all();

    // ==========================================================================
    // Configuration
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void gen_config_set_weights(
        double w_centroid, double w_pmi, double w_attn, double w_global);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void gen_config_set_policy(int greedy, double temperature);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void gen_config_set_filter(UIntPtr max_candidates, double hilbert_range);

    // ==========================================================================
    // Generation
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern UIntPtr gen_generate(
        [MarshalAs(UnmanagedType.LPStr)] string startLabel,
        UIntPtr maxTokens,
        [Out] GenTokenResult[] results);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern UIntPtr gen_find_similar(
        [MarshalAs(UnmanagedType.LPStr)] string label,
        UIntPtr k,
        [Out] GenSimilarResult[] results);

    // ==========================================================================
    // Vocabulary Lookup
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern long gen_vocab_find_label([MarshalAs(UnmanagedType.LPStr)] string label);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern IntPtr gen_vocab_get_label(UIntPtr idx);

    // ==========================================================================
    // Debugging/Testing
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern IntPtr gen_find_similar(
        [MarshalAs(UnmanagedType.LPStr)] string label,
        UIntPtr k,
        out int resultCount);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern IntPtr gen_score_candidates(
        [MarshalAs(UnmanagedType.LPStr)] string currentLabel,
        UIntPtr topK,
        out int resultCount);

    // ==========================================================================
    // Geometric Operations (4D Coordinate System)
    // ==========================================================================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void geom_map_codepoint(
        uint codepoint,
        out Point4D coords);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern double geom_euclidean_distance(
        [In] ref Point4D a,
        [In] ref Point4D b);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void geom_centroid(
        [In] Point4D[] points,
        UIntPtr count,
        out Point4D result);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern void geom_weighted_centroid(
        [In] Point4D[] points,
        [In] double[] weights,
        UIntPtr count,
        out Point4D result);
}

/// <summary>
/// Marshalled structures matching C++ side
/// </summary>
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct GeneratedTokenResult
{
    public UIntPtr token_index;
    public double score_centroid;
    public double score_pmi;
    public double score_attn;
    public double score_global;
    public double score_total;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct SimilarResult
{
    public UIntPtr index;
    public double similarity;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct VocabEntry
{
    [MarshalAs(UnmanagedType.LPStr)]
    public string label;
    public long depth;
    public double frequency;
    public double hilbert_index;
    public Centroid4D centroid;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct Centroid4D
{
    public double x, y, z, m;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct GenTokenResult
{
    public UIntPtr token_index;
    public double score_centroid;
    public double score_pmi;
    public double score_attn;
    public double score_global;
    public double score_total;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct GenSimilarResult
{
    public UIntPtr index;
    public double similarity;
}

/// <summary>
/// 4D point for geometric operations (matches C++ Point4D)
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 8)]
public struct Point4D
{
    public ulong x, y, z, m;

    public Point4D(ulong x = 0, ulong y = 0, ulong z = 0, ulong m = 0)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.m = m;
    }

    public double[] ToArray() => new[] { x / 18446744073709551615.0, y / 18446744073709551615.0, z / 18446744073709551615.0, m / 18446744073709551615.0 };
    public static Point4D FromArray(double[] coords) => new(
        (ulong)(coords[0] * 18446744073709551615.0),
        (ulong)(coords[1] * 18446744073709551615.0),
        (ulong)(coords[2] * 18446744073709551615.0),
        (ulong)(coords[3] * 18446744073709551615.0)
    );
}