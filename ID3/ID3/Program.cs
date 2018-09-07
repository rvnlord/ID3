using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using MoreLinq;

namespace ID3
{
    public class Program
    {
        public static void Main()
        {
            Console.WriteLine("File path: ");
            var path = Console.ReadLine() ?? ""; // path to file containing set of data
            Console.WriteLine("Categorical attribute: ");
            var category = (Console.ReadLine() ?? "").ToLower(); // name of categorical attribute (if names row is not present, then attributes are labeled { x1, x2, ... , xn })

            try
            {
                var dt = ParseFile(File.ReadAllLines(path)); // parse file to data table
                var tree = Learner.ID3(category, dt); // recursively build ID3 tree
                Console.WriteLine();
                PrintTree(tree); // recursively display the tree in console window
                Console.WriteLine($"\nEntropy: {Learner.Entropy(dt.ToColumnsDictionary()[category]):0.000}"); // show entropy of categorical attribute (this is an addition to verify your calculations)
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }

            Console.WriteLine("\nPress any key to quit...");
            Console.ReadKey();
        }

        private static DataTable ParseFile(string[] lines) // Method writing file to data table 
        {
            var dt = new DataTable();
            var firstRow = 0;
            if (lines[1].Distinct().First() == '=') // parse headers if present
            {
                var headers = lines[0].Split(',').Select(w => w.Trim()).ToArray();
                foreach (var colHeader in headers)
                    dt.Columns.Add(colHeader.ToLower());
                firstRow = 2;
            }
            else // add headers if not present
            {
                var nCols = lines[2].Split(',').Length;
                for (var i = 0; i < nCols; i++)
                    dt.Columns.Add($"x{i + 1}");
            }

            for (var i = firstRow; i < lines.Length; i++)
                dt.Rows.Add(lines[i].Split(',').Select(w => w.Trim()).Cast<object>().ToArray());

            return dt;
        } 

        public static void PrintTree(DecisionTree tree, string strTree, int nInd = 0) // Method for recursive printing of trees
        {
            var ind = Enumerable.Repeat(" ", nInd).JoinAsString();
            Console.WriteLine($"{ind}{strTree}");
            foreach (var node in tree.SubNodes)
                PrintTree(node.Value, $"{node.Key} --> {node.Value.Attribute}", nInd + 2);
        } 

        public static void PrintTree(DecisionTree tree) => PrintTree(tree, tree.Attribute);
    }

    public class Learner // Class containing static methods associated with ID3 algorithm
    {
        // This is the simplest version of the algorithm with all the fancy stuff removed, just so you can get through this part of machine learning course as requested, this means that continuus data are NOT supported. If you need more complete solution, don't hestitate to contact me
        public static DecisionTree ID3(string category, DataTable s, Dictionary<string, object[]> possAttrVals = null)
        {
            if (possAttrVals == null) // Only in first run create dictionary where key = attribute name, value = array of possible values. This dictionary is then passed to any recursive step that comes the current one and is not changed during the execution
                possAttrVals = s.ToColumnsDictionary().ToDictionary(kvp => kvp.Key, kvp => kvp.Value.Distinct().ToArray()); 
            if (s.Rows.Count <= 0) // 1. leave recursion if learning set is empty
                return new DecisionTree("(brak danych)");
            var c = s.ToColumnsDictionary()[category]; // values array of categorical attribute
            var r = s.ToColumnsDictionary(category); // dictionary where key = name of conditional attribute, value = array of all conditional attribute values
            if (c.Distinct().Count() <= 1) // 2. leave recursion if all elements of the set have the same value for categorical attrbute
                return new DecisionTree(c[0].ToString());
            var mostFreqCatVal = c.GroupBy(g => g.ToString()).MaxBy(g => g.Count()).Key;
            if (r.Count == 0) // 3. leave recursion if set of conditional attributes is empty
                return new DecisionTree(mostFreqCatVal);

            var dName = r.Select(attr => new { Attribute = attr.Key, Gain = Gain(attr.Value, c) }).MaxBy(a => a.Gain).Attribute; // conditional attribute with largest 'Gain' value        
            var d = r[dName].Distinct().ToList(); // possible values for attribute with largest 'Gain' (excluding those from already used attributes)

            var decisionTree = new DecisionTree(dName); // create new (sub)tree, where node = name of attribute with largest 'Gain' value
            var sSubsets = Partition(s, dName, possAttrVals[dName]); // split subsets of 's' relatively to attribute with largest 'Gain' value

            foreach (var sn in sSubsets) // recursively build subtree for each created subset 
            {
                sn.Value.Columns.Remove(dName); // remove column of values of attribute with largest 'Gain' from subset
                decisionTree.AddSubNode(sn.Key, ID3(category, sn.Value, possAttrVals)); // recursively add subtree for each subset 
                d.RemoveIfExists(sn.Key); // remove name of the node from the array of values of attribute with largest 'Gain' (for which the subtree has been built)
            }

            return decisionTree; // return (sub)tree
        }

        public static double Entropy(object[] set)
        {
            return -set.GroupBy(g => g).Select(g =>
            {
                var p = (double)g.Count() / set.Length;
                return p * Math.Log(p, 2);
            }).Sum();
        }

        public static double AttributeInfo(object[] attribute, object[] category)
        {
            var bothSets = Enumerable.Range(0, attribute.Length).Select((x, i) => new[] { attribute[i], category[i] }).ToArray();

            return attribute.GroupBy(g => g).Select(g =>
            {
                var subG = bothSets.Where(x => Equals(x[0], g.Key)).Select(x => x[1]).ToArray();
                var p = (double)g.Count() / attribute.Length;
                return p * Entropy(subG);
            }).Sum();
        }

        public static double Gain(object[] set, object[] category)
        {
            return Entropy(category) - AttributeInfo(set, category);
        }

        private static Dictionary<string, DataTable> Partition(DataTable s, string bestAttr, object[] bestAttrPossVals)
        {
            var partitions = bestAttrPossVals.ToDictionary(possVal => possVal.ToString(), possVal => s.Clone()); // create dictionary where key = possible value of attribute, value = data table (empty)
			
            foreach (var row in s.Rows) // for each row in the set
            {
                var bestAttrVal = ((DataRow)row)[bestAttr].ToString(); // take attribute value
                partitions[bestAttrVal].Rows.Add(((DataRow)row).ItemArray); // take row and insert it into the data table next to the key denoting taken attribute value
            }

            return partitions;
        }
    }

    public class DecisionTree
    {
        public string Attribute { get; set; }
        public Dictionary<string, DecisionTree> SubNodes { get; } = new Dictionary<string, DecisionTree>();
        public DecisionTree(string attr) => Attribute = attr;
        public void AddSubNode(string val, DecisionTree decisionTree) => SubNodes.Add(val, decisionTree);
    }

    public static class Extensions // additional extension methods
    {
        public static bool EqualsAny<T>(this T o, params T[] os) // verify if value is equal to any value in the supplied list
        {
            foreach (var s in os)
            {
                if (Equals(o, s))
                    return true;
            }
            return false;
        }

        public static Dictionary<string, object[]> ToColumnsDictionary(this DataTable dt, params string[] columnsToSkip) // cast data table to dictionary of columns where key = column header, value = array of cells. Parameter allows to skip unnecessary columns
        {
            var dict = new Dictionary<string, object[]>();
            for (var i = 0; i < dt.Columns.Count; i++)
                if (!dt.Columns[i].ColumnName.EqualsAny(columnsToSkip))
                    dict.Add(dt.Columns[i].ColumnName, dt.Rows.Cast<DataRow>().Select(row => row[dt.Columns[i]]).ToArray());
            return dict;
        }

        public static void RemoveIfExists<T>(this List<T> list, T item) // remove from dictionary element with specified key if it exists
        {
            if (list.Contains(item))
                list.Remove(item);
        }

        public static string JoinAsString<T>(this IEnumerable<T> enumerable, string strBetween = "") // join values of enumerable using specified separator and return it as string
        {
            return string.Join(strBetween, enumerable);
        }

        public static string AsString(this Dictionary<string, DataTable> dict)
        {
            var sb = new StringBuilder();
            foreach (var kvp in dict)
            {
                sb.Append(kvp.Key);
                sb.Append($"\n{kvp.Value.AsString()}");
            }
            return sb.ToString();
        }

        public static string AsString(this DataTable t, string separator = ",")
        {
            var sb = new StringBuilder();
            foreach (DataRow row in t.Rows)
                sb.Append($"  {row.ItemArray.JoinAsString(separator)}\n");
            return sb.ToString();
        }
    } 
}