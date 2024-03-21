import React, { useEffect } from 'react';
import Plot from 'react-plotly.js'

const AttentionVisual = (props) => {
	const {query, attentionData} = props;


	// Assuming `attentionData` is a 2D array passed in as props to this component
	return (
		<Plot
			data={[
				{
					z: attentionData,
					type: 'heatmap',
					colorscale: 'Viridis',
				}
			]}
			layout={{
				title: 'Query Attention Weights Heatmap',
				xaxis: { 
					title: 'Token Index',
					tickmode:'array',
					tickvals: query.map((_, index) => index),
					ticktext: query
				},
				yaxis: { 
					title: 'Token Index',
					tickmode:'array',
					tickvals: query.map((_, index) => index),
					ticktext: query
				},
				width: 300, // Customize the size as needed
				height: 300,
			}}
		/>
	);
};

export default AttentionVisual;