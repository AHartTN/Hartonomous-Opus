$content = Get-Content CMakeLists.txt -Raw
$content = $content -replace 'set\(THREADS_PREFER_PTHREAD_FLAG ON\)', "if(NOT WIN32)`r`n  set(THREADS_PREFER_PTHREAD_FLAG ON)`r`nendif()"
Set-Content CMakeLists.txt $content

$content = Get-Content src/CMakeLists.txt -Raw
$content = $content -replace '# Link threads\.\s*target_link_libraries\(benchmark PRIVATE Threads::Threads\)', "# Link threads.`r`nif(NOT WIN32)`r`n  target_link_libraries(benchmark PRIVATE Threads::Threads)`r`nendif()"
$content = $content -replace 'if\(HAVE_LIB_RT\)\s*target_link_libraries\(benchmark PRIVATE rt\)\s*endif\(HAVE_LIB_RT\)', "if(HAVE_LIB_RT AND NOT WIN32)`r`n  target_link_libraries(benchmark PRIVATE rt)`r`nendif()"
Set-Content src/CMakeLists.txt $content