for program in program_list:  # program_list should contain all your program names
    try:
        spark.sql(f"DROP TABLE IF EXISTS {program}_bom_completion_snapshot")
        print(f"Dropped table: {program}_bom_completion_snapshot")
    except Exception as e:
        print(f"Error dropping table {program}_bom_completion_snapshot: {e}")