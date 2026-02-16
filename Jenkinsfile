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
    }

    post {
        always {
            cleanWs()
        }
    }
}
