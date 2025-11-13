let
    // 1. Get all files in folder
    Source = Folder.Files("C:\EVMS\pbi_exports"),

    // 2. Keep only .xlsx files
    #"Filtered Excel" = Table.SelectRows(Source, each Text.EndsWith([Extension], ".xlsx")),

    // 3. Extract file name (no extension) for program/snapshot
    #"Added FileName" = Table.AddColumn(#"Filtered Excel", "FileNameNoExt",
        each Text.BeforeDelimiter([Name], "."), type text),

    // 4. Load each workbook
    #"Added Workbook" = Table.AddColumn(#"Added FileName", "WB",
        each Excel.Workbook([Content], true)),

    #"Expanded WB" = Table.ExpandTableColumn(#"Added Workbook", "WB",
        {"Name", "Data"}, {"SheetName", "Data"}),

    // 5. Filter for cost_performance sheet
    #"Filtered to Cost" = Table.SelectRows(#"Expanded WB",
        each [SheetName] = "cost_performance"),

    // 6. Expand the sheet table
    #"Expanded Data" = Table.ExpandTableColumn(#"Filtered to Cost", "Data",
        Table.ColumnNames(#"Filtered to Cost"[Data]{0})),

    // 7. Derive ProgramID and SnapshotDate from filename: PROGRAM_YYYY-MM-DD.xlsx
    #"Added Program" = Table.AddColumn(#"Expanded Data", "ProgramID",
        each Text.BeforeDelimiter([FileNameNoExt], "_"), type text),

    #"Added Snapshot" = Table.AddColumn(#"Added Program", "SnapshotDate",
        each Date.From(Text.Middle([FileNameNoExt],
            Text.PositionOf([FileNameNoExt], "_") + 1, 10)), type date),

    // 8. Reorder columns so Program/Snapshot/SUB_TEAM are first
    #"Reordered Columns" =
        let
            baseCols = {"ProgramID", "SnapshotDate", "SUB_TEAM"},
            otherCols = List.RemoveItems(Table.ColumnNames(#"Added Snapshot"), baseCols)
        in
            Table.ReorderColumns(#"Added Snapshot", baseCols & otherCols)
in
    #"Reordered Columns"