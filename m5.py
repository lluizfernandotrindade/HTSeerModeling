import pandas as pd
import numpy as np 
import os 
import json
from hts.hierarchy import HierarchyTree

class M5:
    def __init__(self, path, samples=None):
        self.path = path
        self.train = None
        self.calendar = None
        self.train_date_id=None
        self.day_cols=None
        self.hierarchy = None
        self.train_set = None
        self.idx= None
        self.sampleSize= samples
        self.load()

    def load(self):
        calendar_path = ''
        train_validation_path = ''
        for f in os.listdir(self.path):
            if f.find("calendar")!=-1:
                calendar_path=f
            
            if f.find("train_validation")!=-1 :
                train_validation_path = f

        self.load_train(self.path+'/' + train_validation_path) 
        self.load_calendar(self.path+'/' + calendar_path)

        state_ts = self.transpose('state_id')
        store_ts = self.transpose('store_id')
        cat_ts = self.transpose('cat_id')
        dept_ts = self.transpose('dept_id')
        item_ts = self.transpose('id')

        df = pd.concat([state_ts,store_ts,cat_ts,dept_ts,item_ts],1)
        df['total'] = df['CA']+df['TX'] + df['WI']

        self.train_set = df

        self.build_tree()

    def build_tree(self):
        states = self.train.state_id.unique()
        stores = self.train.store_id.unique()
        depts = self.train.dept_id.unique()
        cats = self.train.cat_id.unique()
        items = self.train.id.unique()

        total = {'total': list(states)}
        state_h = {k: [v for v in stores if v.startswith(k)] for k in states}
        store_h = {k: [v for v in cats if v.startswith(k)] for k in stores}
        dept_h = {k: [v for v in depts if v.startswith(k)] for k in cats}
        item_h = {k: [v for v in items if v.startswith(k)] for k in depts}

        hierarchy = {**total,**state_h,**store_h,**dept_h,**item_h}
        self.hierarchy = hierarchy
        
        self.save_as_json(hierarchy)

        ht = HierarchyTree.from_nodes(nodes=hierarchy, df=self.train_set)
        
        
    def save_as_json(self,hierarchy):
        
        with open('data/hierarchy.json','w') as j:
            json.dump(hierarchy,j)


    def load_train(self,path=None):
        
        if path==None:
            self.path = "./data/"
        self.train = pd.read_csv(path,encoding='utf-8', engine='c')
        
        if self.sampleSize:
            self.train = self.train.sample(n=self.sampleSize, random_state=43)
            
        return self.preprocess_train()
        
    def load_calendar(self,path=None):
        
        if path=="./data/":
            self.path_calendar = path
        self.calendar = pd.read_csv(path)
        return self.preprocess_calendar()

    def preprocess_train(self):
        try:
            self.train['cat_id'] = (self.train['store_id']+'_'+self.train['cat_id'])
            self.train['dept_id'] = (self.train['store_id']+'_'+self.train['dept_id'])
            self.train['id'] = (self.train['store_id']+'_'+self.train['id'])
            
        except:
            print("Parece que a função de pré-processamento tem um problema:")
    
    def preprocess_calendar(self):
        self.day_cols = [col for col in self.train.columns if col.startswith('d_')]
        self.idx = [int(col.split('d_')[1]) for col in self.day_cols]
        self.train_date_id = pd.to_datetime(self.calendar[self.calendar.d.apply(lambda x: int(x.split('d_')[1])).isin(self.idx)].date)

    def transpose(self,column):
        ts = []
        new_cols = self.train[column].unique()
    
        for value in new_cols:
            value_ts = self.train[self.train[column] == value]
            vertical = value_ts[self.day_cols].sum().T
            vertical.index = self.train_date_id
            ts.append(vertical)
        return pd.DataFrame({k: v for k, v in zip(new_cols, ts)})

    def get_state(self):
        pass

    def get_stores(self):
        pass

    def get_departments(self):
        pass

    def get_categories(self):
        pass

    def get_itens(self):
        pass

    def del_state(self):
        pass

    def del_store(self):
        pass

    def del_category(self):
        pass

    def del_item(self):
        pass

    def train_test_split(self):
        pass

    def show_ht(self):
        pass
