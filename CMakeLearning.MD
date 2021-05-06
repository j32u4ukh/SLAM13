### include(CTest)

automatically creates a BUILD_TESTING option. After including the module, use code like:

```cmake
if(BUILD_TESTING)
    # CMake code to create tests ...
endif()
```

to creating tests when testing is enabled.

The definition will write into CTestConfig.cmake

### enable_testing()

Enables testing for this directory and below. See also the add_test() command. 

Note that ctest expects to find a test file in the build directory root. 

Therefore, this command should be in the source directory root. 

 調用 add_test(...) 時，除非已調用enable_testing()，否則CMake不會生成測試。
 
 **請注意，您通常不需要直接調用此方法。只需 include(CTest)，它將為您調用它。**

CMake設置通常可如下所示：

// note: this adds a BUILD_TESTING which defaults to ON

```cmake
include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
```

在測試目錄中：
	
// setup test dependencies

// googletest has some code they explain on how to set it up; put that here

```cmake
add_executable(MyUnitTests
    # ...
)

target_link_libraries(MyUnitTests gtest_main)

add_test(MyUnitTestName MyUnitTests)
```

---

### file

This command is dedicated to file and path manipulation requiring access to the filesystem.


#### GLOB / GLOB_RECURSE

```
file(GLOB <variable> 
    [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS] 
    [<globbing-expressions>...])

file(GLOB_RECURSE <variable> [FOLLOW_SYMLINKS]
     [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS]
     [<globbing-expressions>...])
```

Generate a list of files that match the `<globbing-expressions>` and store it into the `<variable>`.

Globbing expressions are similar to regular expressions, but much simpler. 

If `RELATIVE` flag is specified, the results will be returned as relative paths to the given path.

### add_executable

```
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
```

Adds an executable target called `<name>` to be built from the source files listed in the command invocation. 

The `<name>` corresponds to the logical target name and **must be globally unique** within a project. 

The actual file name of the executable built is constructed based on conventions of the native platform (such as `<name>.exe` or just `<name>`).

### CPack

The CPack module generates the configuration files CPackConfig.cmake and CPackSourceConfig.cmake. 

They are intended for use in a subsequent run of the cpack program where they **steer the generation of installers or/and source packages**.

### set

```
set(<variable> <value>... [PARENT_SCOPE])
```

Sets the given `<variable>` in the current function or directory scope.

If the PARENT_SCOPE option is given the variable will be set in the scope above the current scope. 

Each new directory or function creates a new scope.

#### Set Environment Variable

```
set(ENV{<variable>} [<value>])
```

Sets an Environment Variable to the given value. Subsequent calls of `$ENV{<variable>}` will return this new value.

**This command affects only the current CMake process, not the process from which CMake was called, nor the system environment at large, nor the environment of subsequent build or test processes.**

If no argument is given after ENV{<variable>} or if `<value>` is an empty string, then this command will clear any existing value of the environment variable.

### PROJECT_NAME

Name of the project given to the project command.

This is the name given to **the most recently called project() command** in the current directory scope or above. 

To obtain the name of the top level project, see the CMAKE_PROJECT_NAME variable.

### CMAKE_BUILD_TYPE

Specifies the build type on single-configuration generators.

Possible values are `empty`, `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`, ...

### add_library

Add a library to the project using the specified source files.

```
add_library(<name> [STATIC | SHARED | MODULE]
            [EXCLUDE_FROM_ALL]
            [<source>...])
```

Adds a library target called `<name>` to be built from the source files listed in the command invocation. The `<name>` corresponds to the logical target name and **must be globally unique within a project**. 

The actual file name of the library built is constructed based on conventions of the native platform (such as `lib<name>.a` or `<name>.lib`).

STATIC, SHARED, or MODULE may be given to specify the type of library to be created.

* STATIC libraries are archives of object files for use when linking other targets.
* SHARED libraries are linked dynamically and loaded at runtime. 
* MODULE libraries are plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality. 

If no type is given explicitly the type is STATIC or SHARED based on whether the current value of the variable **BUILD_SHARED_LIBS** is ON. 

For SHARED and MODULE libraries the POSITION_INDEPENDENT_CODE target property is set to ON automatically. A SHARED library may be marked with the FRAMEWORK target property to create an macOS Framework.

### target_include_directories

添加 include 的資料夾路徑。

```cmake
# 實際使用範例：將路徑 '${PROJECT_SOURCE_DIR}/include' 添加到 library hello 當中
target_include_directories(hello PUBLIC ${PROJECT_SOURCE_DIR}/include)
```

### add_subdirectory

Add a subdirectory to the build.

```
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```

Adds a subdirectory to the build. The source_dir specifies the directory in which the source CMakeLists.txt and code files are located. 

If it is a relative path it will be evaluated with respect to the current directory (the typical usage), but it may also be an absolute path.

The binary_dir specifies the directory in which to place the output files. If it is a relative path it will be **evaluated with respect to the current output directory**, but it may also be an absolute path. 

If binary_dir is not specified, the value of source_dir, before expanding any relative path, will be used (the typical usage). 

### target_link_libraries

```
target_link_libraries(<target> ... <item>... ...)
```

The named `<target>` must have been created by a command such as **add_executable()** or **add_library()** and must not be an ALIAS target. 

*The `<target>` doesn't have to be defined in the same directory as the target_link_libraries call.*

Each `<item>` may be:

* A library target name

> The generated link line will have the **full path** to the linkable library file associated with the target. 

* A full path to a library file

> The generated link line will normally preserve the full path to the file.

* A plain library name

> The generated link line will ask the linker to search for the library (e.g. foo becomes -lfoo or foo.lib).

* A link flag

> Item names starting with -, but not -l or -framework, are treated as linker flags.

* A generator expression

> A $<...> generator expression may evaluate to any of the above items or to a semicolon-separated list of them. If the ... contains any ; characters, e.g. after evaluation of a ${list} variable, be sure to use an explicitly quoted argument "$<...>" so that this command receives it as a single `<item>`.

* A debug, optimized, or general keyword immediately followed by another `<item>`. 

> The item following such a keyword will be used only for the corresponding build configuration. 

> The debug keyword corresponds to the Debug configuration (or to configurations named in the DEBUG_CONFIGURATIONS global property if it is set). 

> The optimized keyword corresponds to all other configurations. 

> The general keyword corresponds to all configurations, and is purely optional.

### find_package

```
find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [NO_POLICY_SCOPE])
```

Finds and loads settings from an external project. 

`<PackageName>`_FOUND will be set to indicate whether the package was found. 

When the package is found package-specific information is provided through variables and Imported Targets documented by the package itself. 

The QUIET option disables informational messages, including those indicating that the package cannot be found if it is not REQUIRED. 

The REQUIRED option stops processing with an error message if the package cannot be found.

### include_directories

Add include directories to the build.

```
include_directories([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])
```

Add the given directories to those the compiler uses to search for include files. Relative paths are interpreted as relative to the current source directory.

The include directories are added to the INCLUDE_DIRECTORIES directory property for the current CMakeLists file. They are also added to the INCLUDE_DIRECTORIES target property for each target in the current CMakeLists file.

### list

The list subcommands **APPEND**, **INSERT**, **FILTER**, **PREPEND**, **POP_BACK**, **POP_FRONT**, **REMOVE_AT**, **REMOVE_ITEM**, **REMOVE_DUPLICATES**, **REVERSE** and **SORT** may create new values for the list within the current CMake variable scope.

Similar to the set() command, the LIST command creates new variable values in the current scope, even if the list itself is actually defined in a parent scope. 

#### Modification

```
list(APPEND <list> [<element> ...])
```

Appends elements to the list.

### CMAKE_MODULE_PATH

List of directories specifying a search path for CMake modules to be loaded by the include() or find_package() commands before checking the default modules that come with CMake. 

By default it is empty, it is intended to be set by the project.

### add_definitions

Adds -D define flags to the compilation of source files.

```
add_definitions(-DFOO -DBAR ...)
```

Adds definitions to the compiler command line for sources in the current directory and below. 

This command can be used to add any flags, but it is intended to add preprocessor definitions. Flags beginning in -D or /D that look like preprocessor definitions are automatically added to the COMPILE_DEFINITIONS directory property for the current directory. 

Definitions with non-trivial values may be left in the set of flags instead of being converted for reasons of backwards compatibility. 

若 c++ 腳本如下方定義，利用"宏 ENABLE_SSE"來區分是否啟用區塊內的程式碼

```c++
＃ifdef ENABLE_SSE
// ．．．
＃endif
```

則 CMakeLists.txt 當中這麼定義，說明我們要定義 ENABLE_SSE 以進入上方定義區塊。

```cmake
# -D 或 /D 類似保留字，其後才是要定義的對象，例如：ENABLE_SSE
add_definitions("-DENABLE_SSE")
```

甚至可以提供選擇，讓使用者決定是否要使用

```cmake
# 是否要定義 USE_MACRO，預設為 OFF
OPTION(USE_MACRO "Build the project using macro" OFF)

IF(USE_MACRO)
    add_definitions("-DUSE_MACRO")
endif(USE_MACRO)
```



