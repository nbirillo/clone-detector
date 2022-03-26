package org.jetbrains.research.cloneDetector.core.languagescope.java

import com.intellij.psi.PsiMethod
import org.jetbrains.research.cloneDetector.core.structures.IndexedSequence
import org.jetbrains.research.cloneDetector.core.structures.SourceToken
import org.jetbrains.research.cloneDetector.core.utils.asSequence
import org.jetbrains.research.cloneDetector.core.utils.isNoiseElement

class JavaIndexedSequence(val psiMethod: PsiMethod) : IndexedSequence {
    override val sequence: Sequence<SourceToken>
        get() = psiMethod.body?.asSequence()?.filterNot(::isNoiseElement)?.map(::SourceToken) ?: emptySequence()
}
