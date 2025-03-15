import React, { useLayoutEffect } from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Axios from 'axios';

import AttentionVisual from "./AttentionVisual";
import BarChart from "./BarVisual";

import "./Result.css"

import EyeIcon from './../../Assets/eye.png'
import InsightIcon from './../../Assets/insight.png'

function Result(props){
    const {data, index} = props;

    const [showDoc, setShowDoc] = useState(false);
    const [showInsights, setShowInsights] = useState(false);
    const [barData, setBarData] = useState()

    
	useEffect(() => {
        console.log('=============================================')
        let top_attention_tokens = JSON.parse(data.document_top_tokens)

        let sorted_items = Object.entries(top_attention_tokens).sort((a, b) => b[1] -  a[1]);

        const reg_pattern = /^[a-zA-Z0-9]+$/;

        sorted_items = Object.entries(sorted_items).filter(([key, val]) => reg_pattern.test(val[0]) && val[0].length > 2).slice(0,10)

        const final_obj = Object.fromEntries(sorted_items);

        setBarData(final_obj)

    },[])


    return (
        <div className="result-wrapper flex col align-center justify-center">
            <div className="result">
                <h3>{index + 1}</h3>
                <div className="result-info flex col align-start justify-center">
                    <text className="result-name">{data.name}</text>
                    <text>Similarity Score : <text className="highlight">{data.similarity}</text></text>
                </div>
                <div className="flex col align-end">
                    <button className="view-btn" onClick={(e) => setShowDoc(!showDoc)}>
                        Document <img src={EyeIcon}/>
                    </button>
                    <button className="view-btn" onClick={(e) => setShowInsights(!showInsights)}>
                        Insights <img src={InsightIcon}/>
                    </button>
                </div>
            </div>
            {showInsights &&
                <div className="insight-box flex row align-center justify-evenly">
                    <AttentionVisual attentionData={JSON.parse(data.attention)} query={JSON.parse(data.query_tokens)}/>
                    <BarChart data={barData}/>
                </div>
                
                
            }
            {showDoc &&
                <div className="doc-viewer">
                    {data.main}
                </div>
            }
            
        </div>
    )
}

export default Result;