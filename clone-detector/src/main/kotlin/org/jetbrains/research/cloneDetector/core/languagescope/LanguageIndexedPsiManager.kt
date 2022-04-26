package org.jetbrains.research.cloneDetector.core.languagescope

import com.intellij.openapi.components.ServiceManager
import com.intellij.openapi.fileTypes.FileType
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiElement

class LanguageIndexedPsiService {
    private val indexedPsiDefiners = HashMap<String, IndexedPsiDefiner>()

    fun clear() {
        indexedPsiDefiners.clear()
    }

    fun getIndexedPsiDefiner(psiElement: PsiElement): IndexedPsiDefiner? =
        indexedPsiDefiners[psiElement.containingFile?.fileType?.name]

    fun registerNewLanguage(indexedPsiDefiner: IndexedPsiDefiner) {
        indexedPsiDefiners.put(indexedPsiDefiner.fileType, indexedPsiDefiner)
    }

    fun unregisterLanguage(fileType: String) {
        indexedPsiDefiners.remove(fileType)
    }

    fun isFileTypeSupported(fileType: FileType): Boolean {
        return fileType.name in indexedPsiDefiners.values.map { it.fileType }.toSet()
    }
}

val Project.languageSerializer: LanguageIndexedPsiService
    get() = this.getService(LanguageIndexedPsiService::class.java)!!

