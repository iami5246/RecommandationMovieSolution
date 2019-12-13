using Microsoft.ML.Data;
using System.Linq;


using System;
using System.Collections.Generic;
using System.Text;
using System.IO;


namespace MLRecommandMovieConsole
{
    /// <summary>
    /// 사용자별 단일영화등급평점 정보모델
    /// CSV파일내 해당 속성값을 가져올 컬럼의 순번인덱스를 지정한다.
    /// </summary>
    public class MovieRating
    {
        [LoadColumn(0)] public float userId;
        [LoadColumn(1)] public float movieId;
        [LoadColumn(2)] public float Label;
    }

    /// <summary>
    /// 단일영화등급 평점예측 모델
    /// </summary>
    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }

    /// <summary>
    /// 영화정보모델
    /// </summary>
    public class Movie
    {
        public int ID;
        public String Title;
    }

    /// <summary>
    /// 영화목록 모델
    /// </summary>
    public static class Movies
    {
        public static List<Movie> All = LoadMovieData();

        /// <summary>
        /// Get a single movie.`
        /// </summary>
        /// <param name="id">The identifier of the movie to get.</param>
        /// <returns>The Movie instance corresponding to the specified identifier.</returns>        
        public static Movie Get(int id)
        {
            return All.Single(m => m.ID == id);
        }

        public static List<Movie> LoadMovieData()
        {
            string moviesdatasetpath = Path.Combine(Environment.CurrentDirectory, "recommendation-movies.csv");

            var result = new List<Movie>();
            Stream fileReader = File.OpenRead(moviesdatasetpath);
            StreamReader reader = new StreamReader(fileReader);
            try
            {
                bool header = true;
                int index = 0;
                var line = "";
                while (!reader.EndOfStream)
                {
                    if (header)
                    {
                        line = reader.ReadLine();
                        header = false;
                    }
                    line = reader.ReadLine();
                    string[] fields = line.Split(',');
                    int movieId = Int32.Parse(fields[0].ToString().TrimStart(new char[] { '0' }));
                    string movieTitle = fields[1].ToString();
                    result.Add(new Movie() { ID = movieId, Title = movieTitle });
                    index++;
                }
            }
            finally
            {
                if (reader != null)
                {
                    reader.Dispose();
                }
            }

            return result;
        }

    }


}
