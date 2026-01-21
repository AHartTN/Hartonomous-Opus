using System;
using System.Runtime.InteropServices;

public static class GenerativeInterop
{
    private const string DllName = "hypercube_generative";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "geom_map_codepoint")]
    public static extern Point4D geom_map_codepoint(uint codepoint);

    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct Point4D
    {
        public uint x, y, z, m;
        public override string ToString() => $"({x}, {y}, {z}, {m})";
    }
}

class Program
{
    static void Main()
    {
        try
        {
            Console.WriteLine("Mapping 'A'...");
            var pA = GenerativeInterop.geom_map_codepoint('A');
            Console.WriteLine($"A: {pA}");

            Console.WriteLine("Mapping 'a'...");
            var pa = GenerativeInterop.geom_map_codepoint('a');
            Console.WriteLine($"a: {pa}");

            Console.WriteLine("Mapping 'B'...");
            var pB = GenerativeInterop.geom_map_codepoint('B');
            Console.WriteLine($"B: {pB}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
        }
    }
}
