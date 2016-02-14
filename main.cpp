#include <iostream>
//#include <armadillo>
#include <zlib.h>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/time_duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/date_time/local_time_adjustor.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
//#include<xgboost/base.h>
#include<xgboost/c_api.h>
//#include<xgboost/data.h>
//#include<xgboost/feature_map.h>
//#include<xgboost/gbm.h>
#include<xgboost/learner.h>
//#include<xgboost/logging.h>
//#include<xgboost/metric.h>
//#include<xgboost/objective.h>
#include<xgboost/tree_model.h>
//#include<xgboost/tree_updater.h>

using namespace std;
using namespace boost;
using namespace xgboost;


void xgb_test(){
    string cache_prefix,data_file;
    unsigned part_index = 0, num_parts = 1;
    dmlc::Parser<unsigned int>* parser; //(Parser::Create(data_file.c_str(), part_index, num_parts,"double"));
    std::unique_ptr<DataSource> source_p;
    dmlc::DataIter<RowBatch>* iter;
    //   source_p->Load();
    //     std::unique_ptr<xgboost::data::SimpleCSRSource> source_p(new xgboost::data::SimpleCSRSource());
       DMatrix* data22 = DMatrix::Create(parser);
 //    DMatrix* data22 = DMatrix::Create(std::move(source_p));
    vector<DMatrix*> cache_data;//(1,data22);
    Learner* learner22 = Learner::Create(cache_data);
    //  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create("dump.raw.txt", "r"));
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create("/Win_1TB/Ketchum_Code/xgb.model", "r"));
    //  dmlc::istream fi_1(fi.get());
    learner22->Load(fi.get());
    size_t n_indicators = 3;//  clusters[0].size();
    double vol_minute = 0.1;
    vector<double> indicator_values(n_indicators);
    dmlc::DataIter<RowBatch>* dataRow = data22->RowIterator();
    default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,vol_minute);
    vector<float> prediction;
    size_t n_time_stamps = 100;
    for (size_t time_stamp_index = 0; time_stamp_index < n_time_stamps; time_stamp_index++) {
        for (size_t indicator_index = 0; indicator_index < n_indicators; indicator_index++) {
            indicator_values[indicator_index] =  distribution(generator);
        }
        //   DMatrix data(indicator_values);
        learner22->Predict(data22,true,&prediction);
        cout << prediction[0] << endl;
    }


}


int main() {
    xgb_test();
    return 0;
}