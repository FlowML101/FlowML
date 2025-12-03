import { useState } from 'react';
import { Card } from '../components/ui/Card';

export default function ModelLab() {
  const [selectedModel, setSelectedModel] = useState('');
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const availableModels = [
    { id: 'model_churn_v2', name: 'Customer Churn Predictor v2', type: 'Classification', accuracy: '94.2%' },
    { id: 'model_fraud_v1', name: 'Fraud Detection v1', type: 'Classification', accuracy: '89.7%' },
    { id: 'model_sales_v3', name: 'Sales Forecaster v3', type: 'Regression', accuracy: '92.8%' },
    { id: 'model_sentiment_v1', name: 'Sentiment Analyzer v1', type: 'Classification', accuracy: '87.5%' }
  ];

  const mockFeatures = [
    { name: 'age', type: 'number', description: 'Customer age' },
    { name: 'tenure_months', type: 'number', description: 'Account tenure in months' },
    { name: 'monthly_charges', type: 'number', description: 'Monthly subscription cost' },
    { name: 'total_charges', type: 'number', description: 'Total lifetime charges' },
    { name: 'contract_type', type: 'select', options: ['Month-to-month', 'One year', 'Two year'], description: 'Contract type' },
    { name: 'payment_method', type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], description: 'Payment method' }
  ];

  const handlePredict = () => {
    setIsLoading(true);
    setTimeout(() => {
      setPrediction({
        result: 'High Risk',
        confidence: 0.8723,
        probability: { churn: 0.8723, retain: 0.1277 },
        factors: [
          { feature: 'tenure_months', impact: -0.34, direction: 'negative' },
          { feature: 'monthly_charges', impact: 0.28, direction: 'positive' },
          { feature: 'contract_type', impact: -0.22, direction: 'negative' }
        ]
      });
      setIsLoading(false);
    }, 1200);
  };

  return (
    <div className="page-content">
      <div className="page-header" style={{ marginBottom: '32px' }}>
        <div>
          <h1>Model Lab</h1>
          <p className="page-subtitle">Test trained models with custom inputs</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <Card variant="teal" decoration decorationSize="small">
            <div className="card-header">
              <h3>Select Model</h3>
            </div>
            <div className="card-body" style={{ padding: '24px' }}>
              <div style={{ display: 'grid', gap: '14px' }}>
                {availableModels.map((model) => (
                  <div
                    key={model.id}
                    onClick={() => setSelectedModel(model.id)}
                    style={{
                      padding: '14px 16px',
                      background: selectedModel === model.id 
                        ? 'linear-gradient(135deg, rgba(92,124,250,.25) 0%, rgba(76,110,245,.15) 100%)'
                        : 'rgba(255,255,255,.04)',
                      border: selectedModel === model.id
                        ? '1px solid rgba(92,124,250,.3)'
                        : '1px solid rgba(255,255,255,.08)',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text)' }}>{model.name}</div>
                      {selectedModel === model.id && (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#5c7cfa" strokeWidth="2.5">
                          <polyline points="20 6 9 17 4 12"/>
                        </svg>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: '12px', fontSize: '11px', color: 'rgba(255,255,255,.5)' }}>
                      <span>{model.type}</span>
                      <span>•</span>
                      <span>Accuracy: {model.accuracy}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          <Card variant="blue">
            <div className="card-header">
              <h3>Input Features</h3>
            </div>
            <div className="card-body" style={{ padding: '24px' }}>
              <div style={{ display: 'grid', gap: '18px' }}>
                {mockFeatures.map((feature) => (
                  <div key={feature.name}>
                    <label style={{ display: 'block', fontSize: '12px', color: 'rgba(255,255,255,.7)', marginBottom: '6px' }}>
                      {feature.name.replace('_', ' ')}
                      <span style={{ fontSize: '11px', color: 'rgba(255,255,255,.4)', marginLeft: '6px' }}>• {feature.description}</span>
                    </label>
                    {feature.type === 'select' ? (
                      <select
                        value={inputValues[feature.name] || ''}
                        onChange={(e) => setInputValues({ ...inputValues, [feature.name]: e.target.value })}
                        style={{
                          width: '100%',
                          padding: '10px 12px',
                          background: 'rgba(255,255,255,.06)',
                          border: '1px solid rgba(255,255,255,.12)',
                          borderRadius: '6px',
                          color: 'var(--text)',
                          fontSize: '13px'
                        }}
                      >
                        <option value="">Select {feature.name.replace('_', ' ')}</option>
                        {feature.options?.map((opt) => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        value={inputValues[feature.name] || ''}
                        onChange={(e) => setInputValues({ ...inputValues, [feature.name]: e.target.value })}
                        placeholder={`Enter ${feature.name.replace('_', ' ')}`}
                        style={{
                          width: '100%',
                          padding: '10px 12px',
                          background: 'rgba(255,255,255,.06)',
                          border: '1px solid rgba(255,255,255,.12)',
                          borderRadius: '6px',
                          color: 'var(--text)',
                          fontSize: '13px'
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>

              <button 
                onClick={handlePredict}
                disabled={!selectedModel || isLoading}
                className="btn primary"
                style={{ width: '100%', marginTop: '24px', justifyContent: 'center' }}
              >
                {isLoading ? 'Predicting...' : 'Run Prediction'}
              </button>
            </div>
          </Card>
        </div>

        <div>
          <Card variant="purple" decoration>
            <div className="card-header">
              <h3>Prediction Results</h3>
            </div>
            <div className="card-body" style={{ padding: '24px' }}>
              {!prediction ? (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: 'rgba(255,255,255,.4)' }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ margin: '0 auto 16px' }}>
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 16v-4"/>
                    <path d="M12 8h.01"/>
                  </svg>
                  <p style={{ fontSize: '13px' }}>Select a model and provide inputs to generate predictions</p>
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '28px' }}>
                  <div style={{ 
                    padding: '24px', 
                    background: 'linear-gradient(135deg, rgba(139,92,246,.15) 0%, rgba(124,76,230,.08) 100%)',
                    borderRadius: '12px',
                    border: '1px solid rgba(139,92,246,.2)',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.5)', marginBottom: '8px' }}>Predicted Class</div>
                    <div style={{ fontSize: '36px', fontWeight: 700, color: 'var(--text)', marginBottom: '6px' }}>{prediction.result}</div>
                    <div style={{ fontSize: '13px', color: 'rgba(255,255,255,.6)' }}>
                      Confidence: <span style={{ fontWeight: 600, color: '#a78bfa' }}>{(prediction.confidence * 100).toFixed(2)}%</span>
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,.7)', marginBottom: '12px' }}>Class Probabilities</div>
                    <div style={{ display: 'grid', gap: '10px' }}>
                      {Object.entries(prediction.probability).map(([key, value]: [string, any]) => (
                        <div key={key} style={{ 
                          padding: '12px 14px', 
                          background: 'rgba(255,255,255,.04)', 
                          borderRadius: '8px',
                          border: '1px solid rgba(255,255,255,.08)'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text)', textTransform: 'capitalize' }}>{key}</span>
                            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)' }}>{(value * 100).toFixed(2)}%</span>
                          </div>
                          <div style={{ 
                            height: '4px', 
                            background: 'rgba(255,255,255,.08)', 
                            borderRadius: '2px', 
                            overflow: 'hidden' 
                          }}>
                            <div style={{ 
                              height: '100%', 
                              width: `${value * 100}%`,
                              background: 'linear-gradient(90deg, #a78bfa 0%, #8b5cf6 100%)',
                              transition: 'width 0.5s ease'
                            }}/>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,.7)', marginBottom: '12px' }}>Feature Importance</div>
                    <div style={{ display: 'grid', gap: '10px' }}>
                      {prediction.factors.map((factor: any) => (
                        <div key={factor.feature} style={{ 
                          padding: '12px 14px', 
                          background: 'rgba(255,255,255,.04)', 
                          borderRadius: '8px',
                          border: '1px solid rgba(255,255,255,.08)'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text)' }}>{factor.feature.replace('_', ' ')}</span>
                            <span style={{ 
                              fontSize: '11px', 
                              fontWeight: 600,
                              color: factor.direction === 'positive' ? '#10b981' : '#ef4444'
                            }}>
                              {factor.impact > 0 ? '+' : ''}{(factor.impact * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
