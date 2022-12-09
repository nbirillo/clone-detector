package org.jetbrains.research.cloneDetector.core.utils

import com.intellij.openapi.application.Application
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.project.ProjectManager
import com.intellij.openapi.roots.ProjectFileIndex
import com.intellij.openapi.roots.TestSourcesFilter
import com.intellij.openapi.util.Computable
import com.intellij.openapi.vfs.VirtualFileManager
import com.intellij.openapi.vfs.newvfs.BulkFileListener
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.psi.PsiMethod
import com.intellij.psi.impl.source.tree.ElementType
import com.intellij.psi.tree.TokenSet
import com.intellij.psi.util.PsiTreeUtil
import com.intellij.psi.util.elementType
import com.jetbrains.python.PyElementTypes

val Application: Application
    get() = ApplicationManager.getApplication()

fun <T> Application.readAction(action: () -> T): T =
    Application.runReadAction(Computable(action))

fun Project.addBulkFileListener(bulkFileListener: BulkFileListener) {
    messageBus.connect().subscribe(VirtualFileManager.VFS_CHANGES, bulkFileListener)
}

val Project.fileIndex: ProjectFileIndex
    get() {
        return ProjectFileIndex.getInstance(this)
    }

fun PsiFile.isTestFile(): Boolean =
    TestSourcesFilter.isTestSources(virtualFile, project)

fun PsiElement.findTokens(filter: TokenSet): Sequence<PsiElement> =
    this.leafTraverse({ it in filter }) { it.children.asSequence() }

operator fun TokenSet.contains(element: PsiElement?): Boolean = this.contains(element?.node?.elementType)

fun PsiElement.asSequence(): Sequence<PsiElement> =
    this.depthFirstTraverse { it.children.asSequence() }//.filter { it.children.isEmpty() }

private val javaTokenFilter = TokenSet.create(
    ElementType.WHITE_SPACE,
    ElementType.DOC_COMMENT,
    ElementType.C_STYLE_COMMENT,
    ElementType.END_OF_LINE_COMMENT,
//    ElementType.REFERENCE_PARAMETER_LIST,
//    ElementType.MODIFIER_LIST
)

private val pythonTokenFilter = setOf(
    "Py:SINGLE_QUOTED_STRING",
    "Py:DOCSTRING",
    "Py:END_OF_LINE_COMMENT",
    "Py:IMPORT_STATEMENT",
    "Py:FROM_IMPORT_STATEMENT",
//    "Py:RPAR",
//    "Py:LPAR"
// "LBRACE"
// "RBRACE"
)


fun isNoiseElement(psiElement: PsiElement): Boolean = psiElement in javaTokenFilter
    || psiElement.textLength == 0
    || pythonTokenFilter.contains(psiElement.elementType.toString())


fun PsiElement.nextLeafElement(): PsiElement? {
    var current = this
    while (current.nextSibling == null) {
        if (current.parent is PsiFile) return null
        current = current.parent
    }

    current = current.nextSibling
    while (current.firstChild != null)
        current = current.firstChild
    return current
}

fun PsiElement.prevLeafElement(): PsiElement {
    var current = this
    while (current.prevSibling == null)
        current = current.parent
    current = current.prevSibling
    while (current.lastChild != null)
        current = current.lastChild
    return current
}

fun PsiElement.firstEndChild(): PsiElement {
    var current = this
    while (current.firstChild != null) current = current.firstChild
    return current
}

fun PsiElement.lastEndChild(): PsiElement {
    var current = this
    while (current.lastChild != null) current = current.lastChild
    return current
}

val PsiElement?.method: PsiMethod?
    get() =
        if (this is PsiMethod) {
            this
        } else {
            PsiTreeUtil.getParentOfType(this, PsiMethod::class.java)
        }

val CurrentProject: Project?
    get() =
        ProjectManager.getInstance().openProjects.firstOrNull()?.run {
            if (isDisposed) null
            else this
        }

val Project.toolWindowManager: ToolWindowManager
    get() = ToolWindowManager.getInstance(this)
