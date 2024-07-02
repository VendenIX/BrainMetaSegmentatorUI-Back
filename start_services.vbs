Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\start_services.bat" & chr(34), 0
Set WshShell = Nothing