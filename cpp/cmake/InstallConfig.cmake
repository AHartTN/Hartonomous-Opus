# ============================================================================
# Installation Configuration Module
# ============================================================================

if(BUILD_PG_EXTENSION AND TARGET hypercube_c AND TARGET hypercube)
    # The extension DLLs depend on hypercube_c.dll at runtime (Windows loader).
    # Install it into pkglibdir alongside the modules.
    install(TARGETS hypercube_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )

    install(TARGETS hypercube
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )

    install(FILES
        sql/hypercube--1.0.sql
        hypercube.control
        DESTINATION ${PG_SHAREDIR}/extension
    )

    if(TARGET semantic_ops)
        install(TARGETS semantic_ops
            LIBRARY DESTINATION ${PG_PKGLIBDIR}
            RUNTIME DESTINATION ${PG_PKGLIBDIR}
        )
    endif()

    install(FILES
        sql/semantic_ops--1.0.sql
        sql/semantic_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )

    if(TARGET hypercube_ops)
        install(TARGETS hypercube_ops
            LIBRARY DESTINATION ${PG_PKGLIBDIR}
            RUNTIME DESTINATION ${PG_PKGLIBDIR}
        )
    endif()

    install(FILES
        sql/hypercube_ops--1.0.sql
        sql/hypercube_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )

    if(TARGET embedding_c)
        install(TARGETS embedding_c
            LIBRARY DESTINATION ${PG_PKGLIBDIR}
            RUNTIME DESTINATION ${PG_PKGLIBDIR}
        )
    endif()

    if(TARGET embedding_ops)
        install(TARGETS embedding_ops
            LIBRARY DESTINATION ${PG_PKGLIBDIR}
            RUNTIME DESTINATION ${PG_PKGLIBDIR}
        )
    endif()

    install(FILES
        sql/embedding_ops--1.0.sql
        sql/embedding_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )

    if(TARGET generative_c)
        install(TARGETS generative_c
            LIBRARY DESTINATION ${PG_PKGLIBDIR}
            RUNTIME DESTINATION ${PG_PKGLIBDIR}
        )
    endif()

    if(TARGET generative)
        install(TARGETS generative
            LIBRARY DESTINATION ${PG_PKGLIBDIR}
            RUNTIME DESTINATION ${PG_PKGLIBDIR}
        )
    endif()

    install(FILES
        sql/generative--1.0.sql
        sql/generative.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
endif()

# ============================================================================
# Install Core Library Headers (Optional)
# ============================================================================

# This allows other projects to use hypercube_core as a dependency
# Uncomment if you want to install the headers
# install(DIRECTORY include/hypercube
#     DESTINATION include
#     FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
# )

# install(TARGETS hypercube_core
#     EXPORT hypercube-targets
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#     INCLUDES DESTINATION include
# )

# ============================================================================
# Install Tools and CLI (Optional)
# ============================================================================

# Install CLI tools if PostgreSQL client library is available
if(PostgreSQL_FOUND)
    # Install the unified CLI tool
    install(TARGETS hypercube_cli
        RUNTIME DESTINATION bin
    )

    # Install other tools
    install(TARGETS ingest_safetensor extract_embeddings ingest
        RUNTIME DESTINATION bin
    )

    # Install seed_atoms_parallel if available
    if(TARGET seed_atoms_parallel)
        install(TARGETS seed_atoms_parallel
            RUNTIME DESTINATION bin
        )
    endif()

    # Install utility tools
    install(TARGETS model_discovery vocab_extract vocab_ingest
        RUNTIME DESTINATION bin
    )
endif()

# ============================================================================
# Export Configuration (Optional)
# ============================================================================

# This allows other CMake projects to find and use this library
# Uncomment to enable
# include(CMakePackageConfigHelpers)

# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/hypercube-config-version.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY SameMajorVersion
# )

# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hypercube-config.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/hypercube-config.cmake"
#     INSTALL_DESTINATION lib/cmake/hypercube
# )

# install(EXPORT hypercube-targets
#     FILE hypercube-targets.cmake
#     NAMESPACE hypercube::
#     DESTINATION lib/cmake/hypercube
# )

# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/hypercube-config.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/hypercube-config-version.cmake"
#     DESTINATION lib/cmake/hypercube
# )