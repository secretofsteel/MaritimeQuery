import React, { useState, useEffect, useCallback } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, CheckCircle, HelpCircle, MessageSquare } from 'lucide-react';

const FeedbackPanel = ({ tenantId }) => {
  const [analytics, setAnalytics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchAnalytics = useCallback(async () => {
    setIsLoading(true);
    try {
      const url = new URL('/api/v1/admin/feedback-analytics', window.location.origin);
      if (tenantId) url.searchParams.set('target_tenant_id', tenantId);

      const res = await fetch(url.toString());
      if (res.ok) {
        const data = await res.json();
        setAnalytics(data.analytics);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [tenantId]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  if (isLoading) return <div className="p-8 text-center text-gray-500">Loading analytics...</div>;
  if (!analytics || analytics.error) return (
    <div className="p-8 text-center text-gray-500 border border-gray-800 rounded-lg border-dashed">
      {analytics?.error || 'No feedback data available.'}
    </div>
  );

  // Helpers
  const formatPct = (val) => `${Math.round(val || 0)}%`;

  return (
    <div className="h-full flex flex-col bg-gray-900 text-gray-100 overflow-y-auto space-y-6">
      <h2 className="text-xl font-semibold flex items-center gap-2">
        <BarChart3 size={20} /> Feedback Analytics
      </h2>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KPICard 
          title="Total Feedback" 
          value={analytics.total_feedback} 
          icon={MessageSquare}
          color="blue"
        />
        <KPICard 
          title="Satisfaction Rate" 
          value={formatPct(analytics.satisfaction_rate)} 
          icon={CheckCircle}
          color="green" 
        />
        <KPICard 
          title="Incorrect Rate" 
          value={formatPct(analytics.incorrect_rate)} 
          icon={AlertTriangle}
          color="red" 
        />
        <KPICard 
          title="Refined Queries" 
          value={analytics.query_refinement?.total_refined || 0} 
          icon={TrendingUp}
          color="purple" 
        />
      </div>

      {/* Confidence Calibration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="bg-gray-800 p-5 rounded-lg border border-gray-700">
          <h3 className="text-md font-medium text-gray-300 mb-4 flex items-center gap-2">
            <HelpCircle size={18} /> Confidence Calibration
          </h3>
          
          <div className="space-y-4">
            <CalibrationBar 
              label="High Confidence Accuracy" 
              correct={analytics.confidence_calibration.high_conf_accurate}
              wrong={analytics.confidence_calibration.high_conf_wrong}
            />
            <CalibrationBar 
              label="Low Confidence Accuracy" 
              correct={analytics.confidence_calibration.low_conf_accurate}
              wrong={analytics.confidence_calibration.low_conf_wrong}
            />

            <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t border-gray-700">
              <div className="text-center">
                <div className="text-xs text-gray-500 uppercase">Overconfidence</div>
                <div className={`text-xl font-bold ${analytics.confidence_calibration.overconfidence_rate > 20 ? 'text-red-400' : 'text-green-400'}`}>
                  {formatPct(analytics.confidence_calibration.overconfidence_rate)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500 uppercase">Underconfidence</div>
                <div className={`text-xl font-bold ${analytics.confidence_calibration.underconfidence_rate > 30 ? 'text-amber-400' : 'text-green-400'}`}>
                  {formatPct(analytics.confidence_calibration.underconfidence_rate)}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Recommendations */}
        <section className="bg-gray-800 p-5 rounded-lg border border-gray-700">
          <h3 className="text-md font-medium text-gray-300 mb-4 flex items-center gap-2">
            <CheckCircle size={18} /> System Recommendations
          </h3>
          
          <div className="space-y-3">
            {analytics.recommendations && analytics.recommendations.length > 0 ? (
              analytics.recommendations.map((rec, idx) => (
                <div key={idx} className="p-3 bg-gray-900/50 border border-gray-800 rounded flex gap-3 text-sm text-gray-300">
                   <span>{rec}</span>
                </div>
              ))
            ) : (
              <div className="text-gray-500 italic text-sm text-center py-8">
                No active recommendations. System performing within normal parameters.
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

const KPICard = ({ title, value, icon: Icon, color }) => {
  const colors = {
    blue: 'text-blue-400 bg-blue-900/20 border-blue-900',
    green: 'text-green-400 bg-green-900/20 border-green-900',
    red: 'text-red-400 bg-red-900/20 border-red-900',
    purple: 'text-purple-400 bg-purple-900/20 border-purple-900',
  };

  return (
    <div className={`p-4 rounded-lg border ${colors[color].split(' ')[2]} bg-gray-800`}>
      <div className="flex justify-between items-start">
        <div>
          <p className="text-xs text-gray-400 uppercase font-medium">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
        </div>
        <div className={`p-2 rounded-md ${colors[color].split(' ').slice(0, 2).join(' ')}`}>
          <Icon size={20} />
        </div>
      </div>
    </div>
  );
};

const CalibrationBar = ({ label, correct, wrong }) => {
  const total = correct + wrong;
  const correctPct = total ? (correct / total) * 100 : 0;
  
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-500">{total} samples</span>
      </div>
      <div className="h-4 bg-gray-900 rounded-full overflow-hidden flex">
        <div 
          style={{ width: `${correctPct}%` }} 
          className="h-full bg-green-500/70"
          title={`Correct: ${correct}`}
        />
        <div 
          style={{ width: `${100 - correctPct}%` }} 
          className="h-full bg-red-500/70"
          title={`Wrong: ${wrong}`}
        />
      </div>
      <div className="flex justify-between text-[10px] mt-0.5 text-gray-500">
        <span>Correct ({correct})</span>
        <span>Wrong ({wrong})</span>
      </div>
    </div>
  );
};

export default FeedbackPanel;
