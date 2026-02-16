pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("mlops_project:${env.BUILD_NUMBER}")
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    docker.image("mlops_project:${env.BUILD_NUMBER}").inside {
                        sh 'pytest'
                    }
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
