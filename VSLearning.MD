### 右方程式碼縮圖(隨滾輪上下移動)

設定 > 文字編輯器 > 縮圖 > Enable

### header 放到 include 資料夾，VC 找不到檔案

1. 利用 Ctrl + Shift + P 開啟命令列，輸入 setting 來開啟 setting.json。應該會有多個名稱為 setting 的選項，應選擇來源為 /home/j32u4ukh/.config/Code/User/setting.json 的檔案。
2. 添加
```
"C_Cpp.default.includePath": [
        "${default}",
        "${workspaceFolder}/include/"
    ]
```

3. 此時或許會發現，可以編譯通過、也可以正常執行程式，但是 VC 卻一直提示說找不到檔案，"XXX file not found"。可在同一 setting.json 當中添加 
```
"clang.diagnostic.enable": false
```

4. 最終檔案長相如下（上述所添加內容以外，為 VC 原本生成的內容）

```
{
    "fenix.repos": [
        "https://raw.githubusercontent.com/FenixTemplates/Default/master/fenix.json",
        "https://raw.githubusercontent.com/FenixTemplates/EasyCppProjects/master/fenix.json"
    ],
    "fenix.runCommands": "ask",
    "editor.fontSize": 18,
    "window.zoomLevel": 1,
    "cmake.configureOnOpen": true,
    "C_Cpp.default.includePath": [
        "${default}",
        "${workspaceFolder}/include/"
    ],
    "clang.diagnostic.enable": false
}
```

### 同步 setting.json 設定

/home/j32u4ukh/.config/Code/User/setting.json 定義了全域的設定，表示不同專案都可共享相同的設定。

但該設定必須手動令它同步到各個專案當中，其設定才會發揮作用。

使用 Ctrl + Shift + P 開啟命令列，輸入 setting，選擇 Open User Setting，在搜尋欄打入 include 篩選，
延伸模組當中的 C/C++ 的 C_CPP> Default: Include path 選項的左邊有小齒輪，利用該齒輪來同步設定。

接著應再次建制，應該就可使用自己定義的 include path 了。