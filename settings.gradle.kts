rootProject.name = "cloneDetector"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        maven(url = "https://nexus.gluonhq.com/nexus/content/repositories/releases")
    }
}
include("union-find-set")
include("small-suffix-tree")
include("clone-detector")
