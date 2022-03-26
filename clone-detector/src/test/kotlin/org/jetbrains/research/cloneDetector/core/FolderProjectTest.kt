package org.jetbrains.research.cloneDetector.core

import com.intellij.psi.PsiDirectory
import com.intellij.testFramework.fixtures.BasePlatformTestCase
import org.junit.Ignore
import kotlin.properties.Delegates
import kotlin.reflect.KFunction

@Ignore
open class FolderProjectTest(private val testFolder: String) : BasePlatformTestCase() {
    var baseDirectoryPsi by Delegates.notNull<PsiDirectory>()

    override fun getTestDataPath() = testFolder

    override fun setUp() {
        super.setUp()
        val directory = myFixture.copyDirectoryToProject("/", "")
        baseDirectoryPsi = myFixture.psiManager.findDirectory(directory)!!
    }

    companion object {
        // We can not get the root of the class resources automatically
        private const val resourcesRoot: String = "data"

        fun getResourcesRootPath(
            cls: KFunction<FolderProjectTest>,
            resourcesRootName: String = resourcesRoot
        ): String =
            cls.javaClass.getResource(resourcesRootName)?.path ?: error("Resource $resourcesRootName does not exist")
    }
}
