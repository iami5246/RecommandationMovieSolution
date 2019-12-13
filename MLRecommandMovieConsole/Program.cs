using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;


using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace MLRecommandMovieConsole
{
    public class Program
    {
        private static string trainingDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-test.csv");

        static void Main(string[] args)
        {
            //ML.NET의 머신러닝모델 기본 개발과정(파이프라인) : 
            //데이터로딩-> 데이터변환 및 알고리즘적용계획수립:PipeLine ->기계학습(훈련):Fit->훈련된머신러닝모델생성완료
            //훈련된ML모델 성능평가실시(테스트용데이터):Transform -> 머신러닝 모델 활용하기

            #region Step1: 머신러닝 콘텍스트 생성 및 훈련용/테스트용 데이터 로딩

            //머신러닝 콘텍스트 객체 생성하기
            var mlContext = new MLContext();

            //트레이닝용 데이터 로딩하기
            var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');

            //테스트용 데이터 로딩하기
            var testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            #endregion

            #region Step2: 데이터 처리 파이프라인 생성

            //행렬분해 옵션 정의
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            //데이터처리 파이프라인 생성
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "userId",
                    outputColumnName: "userIdEncoded")
                // step1: userId,movieId 키맵핑 : 각 컬럼을 숫자(벡터)값으로 변환한 결과컬럼으로 행렬화한다.
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "movieId",
                    outputColumnName: "movieIdEncoded")

                // step 2: 행렬인수분해 알고리즘(차원감소기법)을 적용하여 Feature를 추출하고 Feature를 기반으로 해당 Feature와 유사값을 찾아내어 추천값을 찾아낸다.
                .Append(mlContext.Recommendation().Trainers.MatrixFactorization(options)));

            #endregion

            #region Step3: 트레이닝용 데이터를 이용해 학습 시킨후  훈련된 머신러닝 모델 생성

            Console.WriteLine("Training the model...");

            //훈련된 머신러닝 모델 생성
            var model = pipeline.Fit(trainingDataView);

            #endregion

            #region Step4: 생성된 머신러닝모델 테스트 데이터 이용 성능평가 실시

            Console.WriteLine("Evaluating the model...");

            //모델 성능평가 후 예측결과 도출
            var predictions = model.Transform(testDataView);

            //테스트 데이터셋의 모든 사용자와 영화의 평점을 예측처리한다. Lable실제 영역값,Score:예측값
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"  RMSE: {metrics.RootMeanSquaredError:#.##}"); //RMSE 개별예측의 오차값
            Console.WriteLine($"  MAE:   {metrics.MeanAbsoluteError:#.##}"); //평균절대예측오차로 평점(등급)값으로 표시
            Console.WriteLine($"  MSE:   {metrics.MeanSquaredError:#.##}");//평균제곱예측오차 또는 MSE의 제곱근=RMSE 
            Console.WriteLine();

            #endregion

            #region 단일평점예측하기: 특정사용자가 특정영화를 좋아할지에 대한 단일 아이템 평점 예측하기

            Console.WriteLine("Calculating the score for user 6 liking the movie 'GoldenEye'...");

            //예측엔진 생성 CreatePredictionEngine<입력모델,출력모델>(적용ML모델)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            //단일값 예측하기
            var prediction = predictionEngine.Predict(
                new MovieRating()
                {
                    userId = 6,   //6번 사용자
                    movieId = 10  // GoldenEye 영화
                }
            );
            Console.WriteLine($"Score: {prediction.Score}");
            Console.WriteLine();

            #endregion

            #region 평점TOP5영화도출하기: 특정사용자가 좋아할만한 5개 영화 도출하기(평점 높은 순위)


            Console.WriteLine("테스트 데이터 기반 머신러닝모델 예측엔진을 이용 6번 사용자를 위한 TOP5 평점영화 추출하기 ");

            //foreach(var m in Movies.All)
            //{
            //    Console.WriteLine($"{m.ID} - {m.Title}");
            //}

            var top5 = (from m in Movies.All
                        let p = predictionEngine.Predict(
                           new MovieRating()
                           {
                               userId = 6,
                               movieId = m.ID
                           })
                        orderby p.Score descending
                        select (MovieId: m.ID, Score: p.Score)).Take(5);

            //TOP5 예측평점 영화 도출

            foreach (var t in top5)
            {
                //Console.WriteLine($"Score: {t.Score} Movie: {t.MovieId}");
                Console.WriteLine($"  Score:{t.Score}\tMovie: {Movies.Get(t.MovieId)?.Title}");
            }

            Console.ReadLine();

            #endregion

        }

    }
}
