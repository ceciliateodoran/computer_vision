import org.gradle.internal.os.OperatingSystem

tasks.register("dockerBuild") {
    doFirst {
        delete("${projectDir}/build")
        if (OperatingSystem.current().isWindows) {
            exec {
                commandLine("cmd", "/c", "docker build -t ubuntu-opencv_build_streaming .")
                standardOutput = System.out
                errorOutput = System.err
            }
        } else {
            exec {
                commandLine("sh", "-c", "docker build -t ubuntu-opencv_build_streaming .")
                standardOutput = System.out
                errorOutput = System.err
            }
        }
    }
    doLast {
        if (OperatingSystem.current().isWindows) {
            exec {
                commandLine(
                    "cmd", "/c",
                    "docker run -v %cd%\\\\\\\\..:/DCCV -v /DCCV/.gradle" +
                            " -v %cd%\\\\\\\\../.gradle:/tmp/.gradle" +
                            " -p 5555:5555 --name ubuntu-opencv_build-container" +
                            " --rm ubuntu-opencv_build_streaming /bin/bash " +
                            "-c \"GRADLE_USER_HOME=/tmp/.gradle ./gradlew domain:build\""
                )
                standardOutput = System.out
                errorOutput = System.err
            }
        } else {
            exec {
                commandLine(
                    "sh", "-c",
                    "docker run" + " -v \$(pwd)/../:/DCCV -p 5555:5555 --name ubuntu-opencv_build-container" +
                            " --rm ubuntu-opencv_build_streaming /bin/bash " +
                            "-c \"gradle domain:build --info\""
                )
                standardOutput = System.out
                errorOutput = System.err
            }
        }
    }
}