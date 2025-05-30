pipeline {
  agent { label 'team5' }

  environment {
    IMAGE_NAME = "server2-rag-pipeline"
    IMAGE_TAG  = "${env.BUILD_NUMBER}"
    GEMINI_API_KEY = credentials('GEMINI_API_KEY')
    VECTOR_API_URL  = credentials('VECTOR_API_URL')
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
          
          # server2-rag 컨테이너 제거 (새 이름)
          if docker ps -a --filter "name=^/server2-rag$" --format "{{.Names}}" | grep -q "^server2-rag$"; then
            docker rm -f server2-rag
          fi

          # 새 컨테이너 실행 (메트릭 포트 포함)
          docker run -d \
            --name server2-rag \
            --network team5-net \
            -p 8125:8125 \
            -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
            -e VECTOR_API_URL="${VECTOR_API_URL}" \
            -e ENABLE_METRICS=true \
            -v /var/logs/server2_rag:/var/logs/server2_rag \
            ${IMAGE_NAME}:${IMAGE_TAG}
        '''
      }
    }
    stage('Health Check') {
      steps {
        sh '''
          # 컨테이너가 정상적으로 시작될 때까지 대기
          sleep 30
          
          # 헬스체크
          curl -f http://localhost:8125/ || exit 1
          
          # 메트릭 엔드포인트 확인
          curl -f http://localhost:8125/metrics || exit 1
          
          echo "Server2-rag 컨테이너가 정상적으로 시작되었습니다."
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
    success {
      echo "Server2-rag 배포가 성공적으로 완료되었습니다."
      echo "메트릭 확인: http://localhost:8125/metrics"
    }
    failure {
      echo "Server2-rag 배포에 실패했습니다."
      sh '''
        echo "컨테이너 로그 확인:"
        docker logs server2-rag || echo "컨테이너가 존재하지 않습니다."
      '''
    }
  }
}