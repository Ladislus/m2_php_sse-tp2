CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
PROJECT(
    "SSE_TP2"
    VERSION 0.1
    LANGUAGES CXX
)

SET(CMAKE_CXX_STANDARD 14)
ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic)

IF(${CMAKE_BUILD_TYPE} MATCHES Debug)
    # Debug
    MESSAGE(STATUS "Compiling in debug mode")
    # Default options to ON (Debug logs & result check)
    OPTION(VERBOSE "Enable verbose output" ON)
    OPTION(CHECK "Enable result check" ON)
    # Remove compiler optimizations & add debug information
    ADD_COMPILE_OPTIONS(-g -O0)
ELSE()
    MESSAGE(STATUS "Compiling in release mode")
    # Default options to OFF (Debug logs & result check)
    OPTION(VERBOSE "Enable verbose output" OFF)
    OPTION(CHECK "Enable result check" OFF)
    # Add compiler optimizations
    ADD_COMPILE_OPTIONS(-O3)
ENDIF()

# if both are set to ON, SSE has priority over BONUS
# To use naive approach, disable both
OPTION(SSE "Enable SSE" ON)
OPTION(BONUS "Enable SSE" OFF)

IF(VERBOSE)
    MESSAGE(STATUS "Verbose output enabled")
    ADD_COMPILE_OPTIONS(-DVERBOSE)
ENDIF()

IF(CHECK)
    MESSAGE(STATUS "Checking for multiplication result with naive method")
    ADD_COMPILE_OPTIONS(-DCHECK_MUL)
ENDIF()

IF(SSE)
    MESSAGE(STATUS "SSE enabled")
    ADD_COMPILE_OPTIONS(-DSSE)
ELSEIF(BONUS)
    MESSAGE(STATUS "Bonus enabled")
    ADD_COMPILE_OPTIONS(-DBONUS)
ELSE()
    MESSAGE(STATUS "No SSE or BONUS enabled")
ENDIF()

FILE(GLOB SRCS ${CMAKE_SOURCE_DIR}/main.cpp)
ADD_EXECUTABLE(${CMAKE_PROJECT_NAME} ${SRCS})

ADD_CUSTOM_TARGET(
    PERF
    COMMAND perf stat -d ./${CMAKE_PROJECT_NAME}
    DEPENDS ${CMAKE_PROJECT_NAME}
)
