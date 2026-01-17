if(NOT BUILD_PG_EXTENSION)
    return()
endif()

# PG compile defs (Windows vs others)
if(WIN32)
    set(PG_COMPILE_DEFS
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        _WINSOCK_DEPRECATED_NO_WARNINGS
    )
else()
    set(PG_COMPILE_DEFS "")
endif()

# PG include dirs
if(WIN32)
    set(PG_INCLUDE_DIRS
        ${PG_INCLUDEDIR_SERVER}/port/win32_msvc
        ${PG_INCLUDEDIR_SERVER}/port/win32
        ${PG_INCLUDEDIR_SERVER}
        ${PG_INCLUDEDIR}
    )
else()
    set(PG_INCLUDE_DIRS
        ${PG_INCLUDEDIR_SERVER}
        ${PG_INCLUDEDIR}
    )
endif()

function(hc_add_pg_extension EXT_NAME SRC_FILE)
    set(options)
    set(oneValueArgs C_BRIDGE OUTPUT_NAME)
    set(multiValueArgs)
    cmake_parse_arguments(HC_EXT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_library(${EXT_NAME} SHARED "${SRC_FILE}")
    set_source_files_properties("${SRC_FILE}" PROPERTIES LANGUAGE C)

    target_include_directories(${EXT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )

    target_compile_definitions(${EXT_NAME} PRIVATE
        ${PG_COMPILE_DEFS}
        HYPERCUBE_VERSION="${PROJECT_VERSION}"
    )

    set_target_properties(${EXT_NAME} PROPERTIES
        LINKER_LANGUAGE C
        PREFIX ""
        OUTPUT_NAME "${HC_EXT_OUTPUT_NAME}"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${EXT_NAME} PRIVATE
            -Wno-macro-redefined
            -Wno-incompatible-pointer-types
            -Wno-microsoft-include
            -Wno-unused-but-set-variable
            -Wno-pointer-sign
            -Wno-language-extension-token
            -Wno-flexible-array-extensions
        )
    elseif(MSVC)
        target_compile_options(${EXT_NAME} PRIVATE /wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
    endif()

    if(HC_EXT_C_BRIDGE)
        target_link_libraries(${EXT_NAME} PRIVATE ${HC_EXT_C_BRIDGE})
    endif()

    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(${EXT_NAME} PRIVATE ${POSTGRES_LIB})
    endif()
endfunction()

hc_add_pg_extension(hypercube      src/pg/hypercube_pg.c      C_BRIDGE hypercube_c  OUTPUT_NAME "hypercube")
hc_add_pg_extension(semantic_ops   src/pg/semantic_ops_pg.c   C_BRIDGE hypercube_c  OUTPUT_NAME "semantic_ops")
hc_add_pg_extension(hypercube_ops  src/pg/hypercube_ops_pg.c  C_BRIDGE hypercube_c  OUTPUT_NAME "hypercube_ops")
hc_add_pg_extension(embedding_ops  src/pg/embedding_ops_pg.c  C_BRIDGE embedding_c  OUTPUT_NAME "embedding_ops")
hc_add_pg_extension(generative     src/pg/generative_pg.c     C_BRIDGE generative_c OUTPUT_NAME "generative")
