import org.gradle.internal.os.OperatingSystem

tasks.register("buildCMake") {
    onlyIf { OperatingSystem.current().isLinux }

    // output per il task
    outputs.dir(buildDir.resolve("install"))

    doLast {
        buildDir.mkdirs() // crea la directory build

        exec {
            workingDir = buildDir
            commandLine("sh", "-c", """
                mkdir -p . &&
                cmake .. -DBUILD_SHARED_LIBS=OFF -DOpenCV_STATIC=ON &&
                cmake --build . &&
                cmake --install . --prefix=${project.buildDir}/install
            """)
            standardOutput = System.out
            errorOutput = System.err
        }
    }
}
