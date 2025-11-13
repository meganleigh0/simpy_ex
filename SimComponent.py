let
    // 1. Get all files in your pbi_exports folder
    Source = Folder.Files("C:\Users\LGRIFF12\Desktop\EVMS\pbi_exports"),

    // 2. Keep only .xlsx files
    #"Filtered Excel" =
        Table.SelectRows(Source, each Text.EndsWith([Extension], ".xlsx")),

    // 3. Add filename without extension (for ProgramID + SnapshotDate)
    #"Added FileName" =
        Table.AddColumn(
            #"Filtered Excel",
            "FileNameNoExt",
            each Text.BeforeDelimiter([Name], "."),
            type text
        ),

    // 4. Load each workbook (this creates a [Data] column ONCE)
    #"Added Workbook" =
        Table.AddColumn(
            #"Added FileName",
            "WB",
            each Excel.Workbook([Content], true)
        ),

    #"Expanded WB" =
        Table.ExpandTableColumn(
            #"Added Workbook",
            "WB",
            {"Name", "Data"},
            {"SheetName", "Data"}
        ),

    // 5. Keep only the cost_performance sheet from each file
    #"Filtered to Cost" =
        Table.SelectRows(#"Expanded WB", each [SheetName] = "cost_performance"),

    // 6. Expand the sheet table into normal columns
    #"Expanded Data" =
        Table.ExpandTableColumn(
            #"Filtered to Cost",
            "Data",
            Table.ColumnNames(#"Filtered to Cost"[Data]{0}),
            Table.ColumnNames(#"Filtered to Cost"[Data]{0})
        ),

    // 7. Derive ProgramID from filename: PROGRAM_YYYY-MM-DD
    #"Added ProgramID" =
        Table.AddColumn(
            #"Expanded Data",
            "ProgramID",
            each Text.BeforeDelimiter([FileNameNoExt], "_"),
            type text
        ),

    // 8. Derive SnapshotDate from filename: PROGRAM_YYYY-MM-DD
    #"Added SnapshotDate" =
        Table.AddColumn(
            #"Added ProgramID",
            "SnapshotDate",
            each Date.From(
                Text.Middle(
                    [FileNameNoExt],
                    Text.PositionOf([FileNameNoExt], "_") + 1,
                    10
                )
            ),
            type date
        ),

    // 9. Reorder and keep only the columns we care about
    #"Reordered Columns" =
        Table.ReorderColumns(
            #"Added SnapshotDate",
            {"ProgramID", "SnapshotDate", "SUB_TEAM", "YTD", "CTD"}
        ),

    #"Removed Other Columns" =
        Table.SelectColumns(
            #"Reordered Columns",
            {"ProgramID", "SnapshotDate", "SUB_TEAM", "YTD", "CTD"}
        )

in
    #"Removed Other Columns"
