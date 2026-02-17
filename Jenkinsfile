
/*
Dette er vores CI pipeline.
Den:
1. Checker kode ud
2. Bygger Docker image
3. Kører tests
4. Rydder op efter sig selv
*/

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
                sh 'docker build -t mlops_project:${BUILD_NUMBER} .'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'docker run --rm mlops_project:${BUILD_NUMBER} pytest'
            }
        }
        stage('Train new model') {
            steps {
                sh 'docker run --rm mlops_project:${BUILD_NUMBER} python src/train.py'
            }
        }
        stage('Evaluate new model') {
            steps {
                sh 'docker run --rm mlops_project:${BUILD_NUMBER} python src/test.py'
            }
        }
    }

    post {
        always {
            // Clean up Docker images and workspace 
            sh 'docker rmi mlops_project:${BUILD_NUMBER} || true'
            cleanWs()
        }
    }
}
