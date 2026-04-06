<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>보스톤 집값</title>
	<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
	<script src="10.3.js"></script>
</head>
<body>
	<script>
		//1. 과거의 데이터를 준비합니다.
		
		/*
		var 보스톤_원인 = [
		    [0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98],
		    [0.02731,0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,396.9,9.14]
		 ]
		 var 보스톤_결과 = [[24],
			    [21.6]];
		*/
		
		var 원인 = tf.tensor(보스톤_원인); //독립변수
		var 결과 = tf.tensor(보스톤_결과); //종속변수
		
		//2. 모델의 모양을 만듭니다.
		var X = tf.input({shape: [13]}); //독립변수 13개
		var Y = tf.layers.dense({units: 1}).apply(X);//종속변수 1개
		var model = tf.model({inputs: X , outputs: Y});
		
		var compileParam ={
				optimizer: tf.train.adam(),
				loss: tf.losses.meanSquaredError};
		
		model.compile(compileParam);
		
		//3. 데이터로 모델을 학습 시킵니다
		//var fitParam = {epochs: 2000};
		var fitParam = {
			epochs: 2000,
			callbacks:{
				onEpochEnd:function(epoch, logs){
					console.log(
							'epoch 횟수=>',epoch, 
							'MSE=>',logs.loss,						
							'RMSE=>', Math.sqrt(logs.loss)		
							);
				}
			}
		};
		
		model.fit(원인, 결과, fitParam).then(function(result){
			
			//4. 모델을 이용합니다 
			// 4.1 기존 데이터를 이용 (평가)
			var 예측한결과 = model.predict(원인);
			예측한결과.print();
			
			//model.save('downloads://boston-model');
		});
	</script>
</body>
</html>