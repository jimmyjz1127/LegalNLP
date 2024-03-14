import React, { useLayoutEffect } from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Axios from 'axios';

import Navbar from './../Navbar/Navbar'
import Result from './../Result/Result'

import LoadingIcon from './../../Assets/loading.gif'

import "./HomePage.css"

function HomePage(props){

    const [results, setResults] = useState([]);
    const [numResults, setNumResults] = useState(0);
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(0)

    const searchQuery = async () => {
        setLoading(1)
        try {
            const res = await Axios({
                method:'POST',
                withCredentials:true,
                data:{
                    query:query
                },
                url:'http://localhost:5000/search'
            })

            const data = JSON.parse(res.data.data)
            console.log(data)
            setResults(data)
            setNumResults(data.length)
            setLoading(0)
        } catch (err) {
            console.log(err)
            setLoading(0)
        }
    }

    return (
        <div id="home-wrapper" className="flex col align-center justify-center">
            <div id='search-bar-modal' className="flex row align-center">
                <input type='text' placeholder="Enter query..." id="search-bar-input" onChange={(e) => setQuery(e.target.value)}/>
                <button id="search-btn" onClick={(e) => searchQuery()}>Search</button>

            </div>
            <div id='results-wrapper'>
                <div id='results-top-bar' className="flex row align-center justify-between">
                    <h2>Result Summary</h2>
                    <div className="flex row align-center">Number of Results : <div id='result-num'>{numResults}</div></div>
                </div>

                <div id="results-main" className="flex col align-start">
                    {loading ?
                        <div id='loader-wrapper' class="flex col align-center justify-center">
                            <img src={LoadingIcon} id='result-loader'/>
                            <h2>Loading...</h2>
                        </div>:
                        
                        results.map((result, index) => {
                            return (
                                <Result data={result} index={index} key={index}/>
                            );
                        })
                        
                    }
                    
                </div>

                <div id='results-bottom-bar' className="flex row align-center">
                    <button className="default-btn" id="download-btn">Download</button>
                </div>
            </div>
        </div>
    ) ;
}

export default HomePage;