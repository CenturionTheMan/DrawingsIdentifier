﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Xml;
using System.Xml.Linq;

using Log = System.Console;

namespace MyBaseLibrary;

public static class FilesCreatorHelper
{

    /// <summary>
    /// Changes file path extension
    /// </summary>
    /// <param name="str"> base string </param>
    /// <param name="extension"> needed extension </param>
    /// <returns> new string with wanted extension </returns>
    public static string ChangeFileExtension(this string str, string extension)
    {
        int beg = str.LastIndexOf(".");
        int safe = str.LastIndexOf(@"\");

        if (beg >= 0 && beg > safe) str = str.Remove(beg);

        str = str.Insert(str.Length, extension);

        return str;
    }


    public static bool CheckIfFileExists(string filePath, int? argNum = null)
    {
        return File.Exists(filePath);
    }

    public static bool CheckIfDirectoryExists(string filePath, int? argNum = null)
    {
        return Directory.Exists(filePath);
    }


    #region CSV FILES
    public static List<string[]> ReadInputFromCSV(string filePath, char separator)
    {
        filePath = filePath.ChangeFileExtension(".csv");
        if(!CheckIfFileExists(filePath))
            throw new Exception($"File with path [{filePath}] does not exist");

        List<string[]> data = new List<string[]>();

        using (var reader = new StreamReader(filePath))
        {
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine()!;
                var values = line.Split(separator);
                data.Add(values);
            }
        }

        return data;
    }

    public static void CreateCsvFile(List<object[]> data, string fileOutputPath, bool shouldReplace = true, char separator = ';')
    {
        fileOutputPath = fileOutputPath.ChangeFileExtension(".csv");

        var csv = new StringBuilder();

        foreach (var line in data)
        {
            string strLine = "";
            foreach (var word in line)
            {
                strLine += word.ToString() + separator;
            }
            strLine = strLine.Remove(strLine.LastIndexOf(separator));
            csv.AppendLine(strLine);
        }


        try
        {
            if (shouldReplace)
                File.WriteAllText(fileOutputPath, csv.ToString());
            else
                File.AppendAllText(fileOutputPath, csv.ToString());
        }
        catch
        {
            throw new Exception("Something went wrong when CSV file was created, make sure that output file path is correct.");
        }    
    }

    #endregion


    #region XML FILES

    public static List<XElement> GetXElementsFromXmlFile(string xmlFilePath, bool getAllGenerations = false, string nameFilter = "")
    {
        CheckIfFileExists(xmlFilePath);

        try { 
            XDocument xml = XDocument.Load(xmlFilePath);

            IEnumerable<XElement> collection;

            var root = xml.Root!;

            if (getAllGenerations)
            {
                collection = (String.IsNullOrEmpty(nameFilter)) ? root.Descendants() : root.Descendants(nameFilter);
            }
            else
            {
                collection = (String.IsNullOrEmpty(nameFilter)) ? root.Elements() : root.Elements(nameFilter);
            }

            return collection.ToList();
        }
        catch
        {
            throw new Exception($"XMl file with path [{xmlFilePath}] is corrupted");
        }
    }

    public static void CreateEmptyXmlFile(string filePath)
    {
        filePath = filePath.ChangeFileExtension(".xml");

        try
        {
            XmlTextWriter writer = new XmlTextWriter(filePath, System.Text.Encoding.UTF8);
            writer.Formatting = Formatting.Indented;
            writer.WriteStartElement("Root");
            writer.WriteEndElement();
            writer.Flush();
            writer.Close();
        }
        catch {
            throw new Exception("Something went wrong when XML file was created, make sure that output file path is correct."); 
        }
    }

    public static XmlTextWriter CreateXmlFile(string filePath, bool createErrMess = true)
    {
        try
        {
            XmlTextWriter writer = new XmlTextWriter(filePath.ChangeFileExtension(".xml"), System.Text.Encoding.UTF8);
            writer.Formatting = Formatting.Indented;
            return writer;
        }
        catch (Exception e)
        {
            if(!createErrMess)
            {
                Console.WriteLine(e.StackTrace);
                throw;
            }

            throw new Exception("Something went wrong when XML file was created, make sure that output file path is correct.");
        }
    }

    public static void CloseXmlFile(this XmlTextWriter writer)
    {
        writer.Flush();
        writer.Close();
    }

    #endregion
}
