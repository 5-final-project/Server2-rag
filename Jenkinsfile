pipeline {
  agent { label 'team5' }

  environment {
    IMAGE_NAME = "server2-rag-pipeline"
    IMAGE_TAG  = "${env.BUILD_NUMBER}"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }
    stage('Build Docker Image') {
      steps {
        sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
      }
    }
    stage('Deploy Container') {
      steps {
        sh '''
          # legacy container 제거 (이름이 agentic_rag 라면)
          if docker ps -a --filter "name=^/agentic_rag$" --format "{{.Names}}" | grep -q "^agentic_rag$"; then
            docker rm -f agentic_rag
          fi

          # 새 컨테이너 실행
          docker run -d \
            --name agentic_rag \
            -p 8125:8125 \
            -v "${WORKSPACE}/.env":".env" \
            --env-file .env \
            -v /var/logs/server2_rag:/var/logs/server2_rag \
            ${IMAGE_NAME}:${IMAGE_TAG}
        '''
      }
    }
    stage('Cleanup') {
      steps {
        sh "docker image prune -f"
      }
    }
  }

  post {
    always {
      echo "Build #${env.BUILD_NUMBER} finished at ${new Date()}"
    }
  }
}