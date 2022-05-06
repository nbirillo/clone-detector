//package org.jetbrains.research.cloneDetector.core
//import com.intellij.openapi.util.io.FileUtil;
//import com.intellij.psi.PsiFileFactory;
//import com.intellij.psi.codeStyle.LanguageCodeStyleSettingsProvider.createFileFromText
//import com.intellij.util.indexing.FileContentImpl.createFileFromText
//import com.jetbrains.python.PythonFileType
//import java.io.File
//
//fun readFileAsTextUsingInputStream(fileName: String)
//        = File(fileName).inputStream().readBytes().toString(Charsets.UTF_8)
//
//fun doAnalyze() {
//    val testPathString = "/Users/konstantingrotov/Documents/Programming/datasets/Lupa-duplicates/data/notebooks/notebooks0#notebooks0/notebook0.py"
//    val text = readFileAsTextUsingInputStream(testPathString)
//    val file = PsiFileFactory.createFileFromText(testPathString, PythonFileType.INSTANCE, text)
//    println(2)
//}
//
