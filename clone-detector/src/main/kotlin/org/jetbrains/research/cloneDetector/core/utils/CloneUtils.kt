package org.jetbrains.research.cloneDetector.core.utils

import com.intellij.ide.SelectInEditorManager
import com.intellij.openapi.editor.Document
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.psi.PsiElement
import org.jetbrains.research.cloneDetector.core.Edge
import org.jetbrains.research.cloneDetector.core.structures.Clone
import org.jetbrains.research.cloneDetector.core.structures.CloneClass
import org.jetbrains.research.cloneDetector.core.structures.SourceToken

@Suppress("UNCHECKED_CAST")
fun Edge.asSequence(): Sequence<SourceToken> {
    if (isTerminal) {
        throw IllegalArgumentException("You should never invoke this method for terminal edge.")
    } else {
        return (sequence.subList(begin, end + 1) as MutableList<SourceToken>).asSequence()
    }
}

val Clone.textRange: TextRange
    get() = TextRange(firstPsi.textRange.startOffset, lastPsi.textRange.endOffset)

val Clone.hasValidElements: Boolean
    get() = firstPsi.isValid && lastPsi.isValid

fun Clone.printText() {
    println(firstPsi.document.getText(textRange))
}

fun Clone.tokenSequence(): Sequence<PsiElement> =
    generateSequence(firstPsi.firstEndChild()) { it.nextLeafElement() }
        .takeWhile { it.textRange.endOffset <= lastPsi.textRange.endOffset }
        .filterNot(::isNoiseElement)

val Clone.textLength: Int
    get() = lastPsi.textRange.endOffset - firstPsi.textRange.startOffset

val Clone.file: VirtualFile
    get() = firstPsi.containingFile.virtualFile

val Clone.project: Project
    get() = firstPsi.project

val PsiElement.document: Document
    get() = containingFile.viewProvider.document!!

val PsiElement.startLine: Int
    get() = document.getLineNumber(textRange.startOffset) + 1

val PsiElement.endLine: Int
    get() = document.getLineNumber(textRange.endOffset) + 1

val CloneClass.project: Project
    get() = clones.first().project

fun Clone.navigateToSource() {
    SelectInEditorManager.getInstance(project).selectInEditor(
        file,
        textRange.startOffset,
        textRange.endOffset,
        false,
        false
    )
}
