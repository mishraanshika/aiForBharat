# Design Document: AI Infrastructure Layer for India's Kirana Sector

## Overview

This system is an AI Infrastructure Layer designed for India's 12 million+ kirana stores—a hyperlocal demand forecasting platform that works with messy, limited data and understands Bharat's unique consumption patterns. The core innovation is Low-Data Adaptive Learning that bootstraps forecasts using regional demand priors, making AI accessible to stores without clean historical records.

The system addresses a national-scale problem: 95% of India's kirana stores operate on intuition rather than data. Even a 5% efficiency improvement translates to billions in economic impact through reduced waste and increased revenue for India's smallest retailers.

### Key Design Principles

1. **Bharat-First Design**: Built for Indian contexts—festivals, monsoons, cricket matches, bandhs, salary cycles, regional preferences
2. **Low-Data Intelligence**: Operates effectively with sparse data using regional priors and transfer learning
3. **Scalable Architecture**: Designed to serve thousands of stores concurrently with distributed computing
4. **Economic Impact Focus**: Every component tracks measurable socio-economic impact
5. **Mobile-First Accessibility**: WhatsApp integration and mobile-responsive design for smartphone access
6. **Multilingual Support**: Hindi, English, and regional language support

### Technology Stack

- **ML Framework**: scikit-learn for baseline models, XGBoost for gradient boosting, LightGBM for efficient training at scale
- **Time-Series**: statsmodels for ARIMA, Prophet for seasonal decomposition with Indian holiday support
- **Data Processing**: pandas for data manipulation, NumPy for numerical operations, Dask for distributed computing
- **API Integration**: requests library for IMD weather API, cricket APIs, festival calendars
- **Storage**: PostgreSQL with PostGIS for geospatial queries (regional priors), Redis for caching
- **Deployment**: Docker containers, Kubernetes for orchestration, cloud-native auto-scaling
- **Messaging**: Twilio WhatsApp API for alert delivery

## Architecture

### Layered Architecture for National Scale

```
┌─────────────────────────────────────────────────────────────┐
│              Delivery Interface Layer                        │
│  (Dashboard, WhatsApp Bot, Mobile App, CSV Export)          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Decision & Recommendation Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Restock      │  │ Alert        │  │ Decision     │     │
│  │ Calculator   │  │ Manager      │  │ Simulator    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                           │
│  │ Impact       │                                           │
│  │ Estimator    │                                           │
│  └──────────────┘                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│           Event-Aware Adjustment Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Event        │  │ Impact       │  │ Forecast     │     │
│  │ Detector     │  │ Scorer       │  │ Adjuster     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Hybrid Forecasting Engine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Time-Series  │  │ Gradient     │  │ Ensemble     │     │
│  │ Models       │  │ Boosting     │  │ Combiner     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Low-Data     │  │ Regional     │                        │
│  │ Bootstrapper │  │ Prior Engine │                        │
│  └──────────────┘  └──────────────┘                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Feature Engineering Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Bharat       │  │ Temporal     │  │ Geospatial   │     │
│  │ Context      │  │ Features     │  │ Features     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Data Ingestion Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Transaction  │  │ IMD Weather  │  │ Festival     │     │
│  │ Collector    │  │ API Client   │  │ Calendar     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Cricket API  │  │ Event        │  │ Regional     │     │
│  │ Client       │  │ Scraper      │  │ Prior Store  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Storage & Caching Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ PostgreSQL   │  │ Redis Cache  │  │ Model        │     │
│  │ + PostGIS    │  │              │  │ Repository   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Low-Data Bootstrap Flow**: New store → Regional prior lookup → Weighted forecast → Gradual transition to store-specific
2. **Training Pipeline**: Historical transactions → Bharat context features → Hybrid model training → Model repository
3. **Inference Pipeline**: Current state + Bharat context → Feature engineering → Ensemble models → Event adjustment → Recommendations
4. **Impact Tracking Pipeline**: Actual vs predicted → Impact metrics → Regional aggregation → National extrapolation

## Components and Interfaces

### 1. Regional Prior Engine (Core Innovation)

**Responsibility**: Enable forecasting for stores with limited data using regional aggregated demand patterns

**Interface**:
```python
class RegionalPriorEngine:
    def calculate_regional_prior(self, store_location: Location, sku_category: str, radius_km: float) -> RegionalPrior
    def find_similar_stores(self, store: Store, radius_km: float) -> List[Store]
    def aggregate_demand_patterns(self, stores: List[Store], sku_category: str) -> DemandPattern
    def get_fallback_prior(self, state: str, sku_category: str) -> DemandPattern
```

**Implementation Strategy**:
- **Geospatial Clustering**: Use PostGIS to find stores within 5km radius with similar characteristics (size, location type, customer demographics)
- **Demand Aggregation**: Calculate median demand patterns across similar stores to create robust priors
- **Hierarchical Fallback**: If local priors unavailable → state-level → national-level category patterns
- **Similarity Scoring**: Weight stores by similarity score based on: distance, store size, urban/rural classification, income demographics
- **Pattern Extraction**: Extract seasonal patterns, day-of-week patterns, festival multipliers from aggregated data

### 2. Low-Data Bootstrapper

**Responsibility**: Generate forecasts for stores with insufficient historical data

**Interface**:
```python
class LowDataBootstrapper:
    def bootstrap_forecast(self, store: Store, sku: str, days_of_data: int) -> Forecast
    def calculate_data_sufficiency_score(self, days_of_data: int) -> float
    def blend_prior_and_store_data(self, regional_prior: RegionalPrior, store_data: DataFrame, weight: float) -> Forecast
```

**Implementation Strategy**:
- **Data Sufficiency Scoring**: score = min(days_of_data / 90, 1.0) - full confidence at 90 days
- **Weighted Blending**: forecast = (1 - score) * regional_prior + score * store_model
- **Gradual Transition**: Over 60 days, linearly increase weight on store-specific data
- **Confidence Adjustment**: Reduce confidence intervals when using regional priors
- **Explainability**: Tag forecasts with data source (regional/store/blended) for transparency

### 3. Hybrid Forecasting Engine

**Responsibility**: Generate demand predictions using ensemble of models optimized for Indian contexts

**Interface**:
```python
class HybridForecastingEngine:
    def train(self, sales_data: DataFrame, bharat_features: DataFrame) -> ModelMetrics
    def predict_next_day(self, sku: str, features: DataFrame) -> Forecast
    def predict_next_week(self, sku: str, features: DataFrame) -> List[Forecast]
    def get_model_metrics(self) -> ModelMetrics
```

**Implementation Strategy**:
- **Ensemble Components**:
  - ARIMA: Captures linear trends and seasonality
  - XGBoost: Captures non-linear patterns and feature interactions
  - Prophet: Handles Indian holidays and monsoon seasonality
- **Bharat Context Features**:
  - Festival indicators (Diwali, Holi, Eid, regional festivals)
  - Monsoon phase (pre-monsoon, monsoon, post-monsoon)
  - Salary cycle flags (days 1-5, 25-31)
  - Cricket match indicators (India matches, IPL)
  - Day-of-week with Indian weekend patterns
- **Weighted Ensemble**: Assign weights based on historical MAPE per SKU category
- **Model Selection**: Use simpler models (moving average) for < 30 days data

### 4. Event-Aware Adjustment Layer

**Responsibility**: Detect Indian-context events and adjust forecasts accordingly

**Interface**:
```python
class EventAwareAdjustmentLayer:
    def detect_events(self, location: Location, date_range: DateRange) -> List[Event]
    def calculate_impact_score(self, event: Event, sku: str) -> float
    def adjust_forecast(self, base_forecast: Forecast, events: List[Event]) -> Forecast
```

**Implementation Strategy**:
- **Event Sources**:
  - Cricket API: India matches, IPL schedule
  - Weather API: Monsoon alerts, heat waves, cyclones
  - Festival Calendar: National and regional festivals
  - News Scraping: Local events (weddings, rallies, bandhs)
- **Impact Scoring**: Train regression model on (event_type, sku_category, proximity) → demand_multiplier
- **Indian Event Types**:
  - Cricket matches: High impact on snacks, beverages, disposable items
  - Festivals: Category-specific impacts (sweets for Diwali, colors for Holi)
  - Bandhs: Reduced demand + pre-bandh stockpiling
  - Monsoons: Increased demand for umbrellas, raincoats, hot beverages
  - Salary days: Increased overall demand, especially premium items
- **Forecast Adjustment**: adjusted_forecast = base_forecast * (1 + impact_score * proximity_factor)

### 5. Decision and Recommendation Layer

**Responsibility**: Convert forecasts into actionable recommendations with simulation capabilities

**Interface**:
```python
class RestockCalculator:
    def calculate_restock_quantity(self, sku: str, forecast: Forecast, current_inventory: int, lead_time_days: int) -> RestockRecommendation
    def prioritize_skus(self, recommendations: List[RestockRecommendation]) -> List[RestockRecommendation]

class DecisionSimulator:
    def simulate_inventory_change(self, changes: Dict[str, int], horizon_days: int) -> SimulationResult
    def calculate_net_profit_impact(self, simulation: SimulationResult) -> float
```

**Implementation Strategy**:
- **Restock Calculation**: restock = max(0, forecast * (1 + lead_time_days) + safety_stock - current_inventory)
- **Safety Stock**: 20% of forecast demand
- **MOQ Rounding**: Round up to supplier minimum order quantity
- **Priority Scoring**: (forecast - inventory) / forecast * unit_revenue
- **Simulation**: Monte Carlo with 1000 iterations sampling from forecast distribution
- **Impact Metrics**: Revenue, waste cost, stockout loss calculated per iteration

### 6. Impact Estimator (Socio-Economic Value)

**Responsibility**: Calculate and track economic impact at store, regional, and national levels

**Interface**:
```python
class ImpactEstimator:
    def calculate_store_impact(self, store_id: str, period_days: int) -> ImpactMetrics
    def calculate_regional_impact(self, region: str, period_days: int) -> ImpactMetrics
    def extrapolate_national_impact(self, sample_stores: List[Store]) -> NationalImpactProjection
    def calculate_roi(self, store_id: str, system_cost: float) -> float
```

**Implementation Strategy**:
- **Stockout Reduction**: Compare actual stockouts to baseline (estimated from pre-system data or industry average)
- **Waste Reduction**: Track unsold inventory that expired or was discarded
- **Revenue Increase**: Attribute to better stock availability and reduced stockouts
- **Regional Aggregation**: Sum impact across all stores in region
- **National Extrapolation**: (avg_impact_per_store * 12_million_stores) with confidence intervals
- **ROI Calculation**: (revenue_increase + waste_savings) / system_cost

### 7. Alert Manager

**Responsibility**: Generate and deliver alerts via dashboard and WhatsApp

**Interface**:
```python
class AlertManager:
    def check_demand_spike(self, sku: str, forecast: Forecast, historical_avg: float) -> Optional[Alert]
    def deliver_alert(self, alert: Alert, delivery_channels: List[str]) -> None
    def send_whatsapp_message(self, phone_number: str, message: str) -> bool
```

**Implementation Strategy**:
- **Spike Detection**: forecast > 1.4 * 30_day_moving_average
- **Alert Enrichment**: Include likely cause from Event-Aware Layer
- **WhatsApp Integration**: Use Twilio API for message delivery in Hindi/English
- **Message Format**: "🔔 Alert: {sku_name} demand spike expected tomorrow. Stock {quantity} units. Reason: {cause}"
- **Delivery Channels**: Dashboard notification + WhatsApp + SMS fallback

## Data Models

### RegionalPrior
```python
@dataclass
class RegionalPrior:
    region_id: str
    sku_category: str
    demand_pattern: DemandPattern
    store_count: int  # number of stores in aggregation
    confidence_score: float
    last_updated: datetime
```

### DemandPattern
```python
@dataclass
class DemandPattern:
    daily_avg: float
    weekly_seasonality: List[float]  # 7 values for each day of week
    monthly_seasonality: List[float]  # 12 values for each month
    festival_multipliers: Dict[str, float]  # festival_name -> multiplier
    monsoon_multiplier: float
    salary_cycle_multiplier: float
```

### Store
```python
@dataclass
class Store:
    store_id: str
    location: Location
    store_type: str  # urban/rural/semi-urban
    size_category: str  # small/medium/large
    income_demographic: str  # low/middle/high
    days_of_data: int
    active_skus: List[str]
```

### Location
```python
@dataclass
class Location:
    latitude: float
    longitude: float
    state: str
    district: str
    pincode: str
```

### Forecast
```python
@dataclass
class Forecast:
    sku: str
    prediction_date: date
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float
    confidence_score: float
    data_sufficiency_score: float
    data_source: str  # regional_prior/store_specific/blended
    contributing_factors: Dict[str, float]
```

### Event
```python
@dataclass
class Event:
    event_id: str
    event_type: str  # cricket_match, festival, bandh, monsoon_alert, wedding
    location: Location
    start_time: datetime
    end_time: datetime
    impact_radius_km: float
    estimated_attendance: Optional[int]
```

### ImpactMetrics
```python
@dataclass
class ImpactMetrics:
    period_start: date
    period_end: date
    stockout_reduction_percent: float
    waste_reduction_percent: float
    revenue_increase_percent: float
    stockout_reduction_rupees: float
    waste_savings_rupees: float
    revenue_increase_rupees: float
    total_economic_impact_rupees: float
```

### NationalImpactProjection
```python
@dataclass
class NationalImpactProjection:
    sample_size: int
    total_kirana_stores: int  # 12 million
    projected_annual_impact_crores: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    assumptions: List[str]
```

### RestockRecommendation
```python
@dataclass
class RestockRecommendation:
    sku: str
    current_inventory: int
    forecast_demand: float
    recommended_quantity: int
    safety_stock: int
    priority_score: float
    reason: str
    estimated_cost: float
    estimated_revenue: float
```

### SimulationResult
```python
@dataclass
class SimulationResult:
    sku_changes: Dict[str, int]
    horizon_days: int
    iterations: int
    avg_daily_revenue: float
    avg_daily_waste: float
    avg_daily_stockouts: float
    net_profit_change: float
    confidence_interval: Tuple[float, float]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Regional Prior Availability

*For any* store with fewer than 30 days of data, the Regional Prior Engine should return either a regional prior from nearby stores, a state-level prior, or a national-level category prior (never failing to provide a forecast).

**Validates: Requirements 1.1, 1.3**

### Property 2: Data Sufficiency Scoring

*For any* number of days of historical data, the data_sufficiency_score should be between 0 and 1, with 0 for no data and 1.0 for 90+ days of data.

**Validates: Requirements 1.5**

### Property 3: Gradual Transition from Priors

*For any* store transitioning from regional priors to store-specific models, as days_of_data increases from 30 to 90, the weight on store-specific data should monotonically increase.

**Validates: Requirements 1.4**

### Property 4: Forecast Generation with Sufficient Data

*For any* SKU with at least 30 days of historical sales data, the Forecasting Engine should generate a next-day forecast with non-null predicted demand and confidence intervals.

**Validates: Requirements 2.1, 2.4**

### Property 5: Bharat Context Feature Incorporation

*For any* two forecast requests with identical sales history but different festival/monsoon/cricket match data, the predicted demand values should differ, demonstrating that Bharat context features influence predictions.

**Validates: Requirements 2.3**

### Property 6: Festival Calendar Completeness

*For any* date within the next 365 days, the festival calendar should be queryable and return either a festival entry or an empty result for both national and region-specific festivals.

**Validates: Requirements 2.7**

### Property 7: Event Proximity Filtering

*For any* event located more than 2 kilometers from the store, the Event-Aware Adjustment Layer should not calculate an impact score or adjust forecasts for that event.

**Validates: Requirements 3.1**

### Property 8: Indian Event Type Coverage

*For any* detected event of type cricket_match, festival, bandh, monsoon_alert, wedding, rally, or exam, the Event-Aware Adjustment Layer should calculate an impact score.

**Validates: Requirements 3.2**

### Property 9: Event-Based Forecast Adjustment

*For any* event with an impact score above 0.6, the adjusted forecast for affected SKUs should differ from the base forecast.

**Validates: Requirements 3.4**

### Property 10: Cricket Match Auto-Flagging

*For any* major cricket match involving India, the system should automatically flag high-demand categories (snacks, beverages, disposable items) for increased stocking.

**Validates: Requirements 3.6**

### Property 11: Bandh Forecast Adjustment

*For any* announced bandh, the system should adjust forecasts to account for both reduced foot traffic during the bandh and potential pre-bandh stockpiling.

**Validates: Requirements 3.7**

### Property 12: Overstock Prevention

*For any* SKU where current inventory exceeds 1.5 times the predicted demand, the recommended restock quantity should be zero.

**Validates: Requirements 4.2**

### Property 13: Restock Calculation Correctness

*For any* SKU where predicted demand exceeds current inventory, the recommended restock quantity should equal (predicted_demand - current_inventory + safety_stock) where safety_stock equals 0.2 times predicted_demand, rounded up to the nearest MOQ multiple.

**Validates: Requirements 4.1, 4.3, 4.4, 4.5**

### Property 14: Simulation Completeness

*For any* inventory change simulation, the result should include revenue impact, waste cost, stockout loss, and net profit change calculated as (revenue - waste - stockout_loss).

**Validates: Requirements 4.6, 4.7**

### Property 15: Simulation SKU Limit

*For any* simulation request with more than 10 SKUs, the system should either reject the request or process only the first 10 SKUs.

**Validates: Requirements 4.8**

### Property 16: SKU Prioritization

*For any* set of restocking recommendations, SKUs should be ordered by priority_score in descending order, where priority_score reflects stockout risk and revenue impact.

**Validates: Requirements 4.9**

### Property 17: Store-Level Impact Calculation

*For any* store with at least 30 days of actual vs predicted data, the Impact Estimator should calculate stockout reduction, waste reduction, and revenue increase percentages.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 18: Regional Impact Aggregation

*For any* region, the aggregated impact metrics should equal the sum of individual store impacts within that region.

**Validates: Requirements 5.5**

### Property 19: Monthly Impact Reporting

*For any* completed month, the system should generate an impact report showing cumulative metrics in rupees.

**Validates: Requirements 5.6**

### Property 20: ROI Calculation

*For any* store with at least 90 days of system usage, the ROI metric should be calculable as (economic_benefits - system_costs) / system_costs.

**Validates: Requirements 5.7**

### Property 21: National Impact Extrapolation

*For any* sample of stores, the national impact projection should scale the average per-store impact to 12 million stores with confidence intervals.

**Validates: Requirements 5.8**

### Property 22: Demand Spike Alert Triggering

*For any* SKU where the predicted demand exceeds the 30-day moving average by more than 40 percent, a demand spike alert should be generated with all required fields.

**Validates: Requirements 6.1, 6.2**

### Property 23: Alert Timing Constraint

*For any* demand spike alert, the alert timestamp plus 24 hours should be less than or equal to the forecast prediction date.

**Validates: Requirements 6.3**

### Property 24: Correlated Alert Grouping

*For any* set of demand spike alerts where the affected SKUs show correlated patterns, the alerts should be grouped with contextual information.

**Validates: Requirements 6.4**

### Property 25: WhatsApp Alert Delivery

*For any* alert with WhatsApp as a delivery channel, the system should attempt to send a message via the WhatsApp API.

**Validates: Requirements 6.5**

### Property 26: IMD Weather Integration

*For any* weather data request, the system should retrieve data from Indian Meteorological Department API or equivalent, including monsoon predictions.

**Validates: Requirements 7.1**

### Property 27: Salary Cycle Identification

*For any* date, if the day-of-month is in the range [1, 5] or [25, 31], the system should flag it as a salary cycle day.

**Validates: Requirements 7.3**

### Property 28: Cricket Schedule Integration

*For any* date within the next 30 days, the system should have access to cricket match schedules for Indian national team and IPL matches.

**Validates: Requirements 7.4**

### Property 29: Synthetic Data Generation Fallback

*For any* SKU with zero historical transactions, when a forecast is requested, the system should generate synthetic training data based on regional patterns rather than failing.

**Validates: Requirements 7.5**

### Property 30: Multilingual SKU Support

*For any* SKU name in Hindi or regional languages, the system should process it equivalently to English SKU names.

**Validates: Requirements 7.7**

### Property 31: Train-Test Split Correctness

*For any* model training run with at least 14 days of data, the test set should contain exactly the most recent 14 days of transactions.

**Validates: Requirements 8.2**

### Property 32: Model Metrics Logging

*For any* completed model training run, the system should log MAPE and RMSE metrics calculated on the test set.

**Validates: Requirements 8.3**

### Property 33: Model Rollback on Performance Degradation

*For any* model retraining where the new model's MAPE is more than 5 percentage points worse than the current model's MAPE, the system should retain the current model.

**Validates: Requirements 8.4**

### Property 34: Prior-to-Store Transition Weight

*For any* store transitioning from regional priors to store-specific models, the weight on store-specific data should increase linearly over 60 days.

**Validates: Requirements 8.6**

### Property 35: Concurrent Store Processing

*For any* batch of up to 1000 stores, the system should generate forecasts concurrently without exceeding 5 seconds per store latency.

**Validates: Requirements 9.1**

### Property 36: Regional Prior Caching

*For any* frequently accessed regional prior, the system should cache it in Redis to minimize database queries and reduce latency.

**Validates: Requirements 9.3**

### Property 37: Auto-Scaling Trigger

*For any* system state where load exceeds 80 percent capacity, the system should trigger auto-scaling to add compute resources.

**Validates: Requirements 9.5**

### Property 38: Forecast Generation Latency

*For any* single store forecast request, the system should complete generation within 5 seconds even under peak load.

**Validates: Requirements 9.6**

### Property 39: Dashboard Data Completeness

*For any* daily dashboard view, it should display forecasts, restocking recommendations, and active alerts for all active SKUs.

**Validates: Requirements 10.1**

### Property 40: Confidence Indicator Display

*For any* displayed forecast, the system should use visual indicators distinguishing high-confidence from low-confidence predictions.

**Validates: Requirements 10.2**

### Property 41: Weekly Report Generation

*For any* completed week, the system should generate a summary report with forecast accuracy, actual vs predicted demand, and revenue impact.

**Validates: Requirements 10.3**

### Property 42: Mobile Responsiveness

*For any* dashboard page, it should render correctly on smartphone screens with mobile-responsive design.

**Validates: Requirements 10.6**

### Property 43: Multilingual Interface

*For any* user interface element, it should be available in both Hindi and English, with support for regional languages.

**Validates: Requirements 10.7**

## Error Handling

### Input Validation Errors

**Missing or Insufficient Data**:
- If sales data < 30 days: Use regional priors automatically (no error, graceful degradation)
- If SKU not found: Return error with message "SKU {sku_id} not found in system"
- If IMD weather API unavailable: Log warning, use cached weather data, continue with degraded forecast
- If regional priors unavailable: Fall back to state-level then national-level priors

**Invalid Input Formats**:
- If CSV missing required columns: Return error listing missing columns
- If date formats invalid: Return error with expected format (ISO 8601)
- If numeric values out of range: Return error specifying valid ranges
- If Hindi/regional language SKU names malformed: Attempt transliteration, log warning if fails

**Business Logic Violations**:
- If inventory < 0: Return error "Inventory cannot be negative"
- If forecast horizon > 30 days: Return error "Maximum forecast horizon is 30 days"
- If simulation SKU count > 10: Truncate to first 10 SKUs and log warning

### Runtime Errors

**Model Training Failures**:
- If training fails due to data quality: Log detailed error, retain previous model, alert administrator
- If training fails due to resource constraints: Retry with reduced batch size, schedule retry for off-peak hours
- If regional prior calculation fails: Fall back to state/national priors, log error

**External API Failures**:
- IMD Weather API timeout: Use cached data from previous successful call, log warning
- Cricket API failure: Continue without cricket match adjustments, log warning
- WhatsApp API failure: Fall back to SMS, then dashboard notification only
- Event scraping failure: Continue with scheduled events only (festivals, cricket), log warning

**Database Errors**:
- PostgreSQL connection failure: Retry 3 times with exponential backoff, if all fail return error to user
- PostGIS geospatial query timeout: Expand search radius, if still fails use state-level priors
- Redis cache miss: Query PostgreSQL directly, repopulate cache
- Constraint violation: Return error with specific constraint violated

### Data Quality Issues

**Anomalous Data Detection**:
- If transaction quantity > 1000x historical average: Flag for manual review, exclude from training
- If price changes > 10x overnight: Flag as potential data error, require confirmation
- If missing data > 20% for a day: Interpolate using adjacent days, mark forecast as low confidence

**Handling Sparse Data**:
- If SKU has < 10 transactions total: Use category-level regional prior
- If recent data missing: Use last-observation-carried-forward for up to 3 days
- If seasonal patterns unclear: Fall back to simpler moving average model

**Regional Prior Quality**:
- If fewer than 3 stores in region: Expand radius to 10km
- If still insufficient: Use state-level priors
- If state-level unavailable: Use national category priors
- Always tag forecast with data source for transparency

## Testing Strategy

### Dual Testing Approach

The system requires both unit testing and property-based testing for comprehensive coverage:

- **Unit tests** verify specific examples, edge cases, and error conditions
- **Property tests** verify universal properties across all inputs
- Both approaches are complementary and necessary

### Unit Testing Focus

Unit tests should focus on:
- Specific examples demonstrating correct behavior (e.g., known forecast scenarios with Indian festivals)
- Edge cases (e.g., empty datasets, single-SKU stores, extreme monsoon conditions, bandh scenarios)
- Error conditions (e.g., invalid CSV formats, API failures, negative inventory)
- Integration points between components (e.g., regional prior → forecast, event detection → adjustment)
- Indian context handling (e.g., Diwali demand spikes, cricket match impacts, salary cycle patterns)

Avoid writing too many unit tests for scenarios that property tests can cover through randomization.

### Property-Based Testing Configuration

**Framework Selection**: Use `hypothesis` library for Python

**Test Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `# Feature: kirana-demand-forecasting, Property {number}: {property_text}`

**Property Test Implementation**:
- Each correctness property listed above must be implemented as a single property-based test
- Tests should generate random but valid inputs (sales data, forecasts, events, stores, locations)
- Tests should verify the property holds across all generated inputs
- Include Indian context in generators (festivals, monsoon dates, cricket matches)

**Example Property Test Structure**:
```python
from hypothesis import given, strategies as st

# Feature: kirana-demand-forecasting, Property 12: Overstock Prevention
@given(
    current_inventory=st.integers(min_value=100, max_value=1000),
    predicted_demand=st.integers(min_value=1, max_value=500)
)
def test_overstock_prevention(current_inventory, predicted_demand):
    """For any SKU where current inventory exceeds 1.5x predicted demand,
    recommended restock should be zero."""
    
    # Assume current_inventory > 1.5 * predicted_demand
    if current_inventory <= 1.5 * predicted_demand:
        return  # Skip this test case
    
    restock_qty = calculate_restock_quantity(
        predicted_demand=predicted_demand,
        current_inventory=current_inventory,
        lead_time_days=1
    )
    
    assert restock_qty == 0, \
        f"Expected 0 restock for overstock situation, got {restock_qty}"
```

### Integration Testing

**End-to-End Scenarios**:
- Complete low-data bootstrap flow: New store → Regional prior → Forecast → Recommendations
- Complete forecast pipeline: Raw data → Bharat features → Hybrid model → Event adjustment → Recommendations
- Event detection → Forecast adjustment → Alert generation (test with Diwali, cricket match, bandh)
- Impact tracking → Regional aggregation → National extrapolation

**External Integration Tests**:
- IMD Weather API integration with mock responses
- Cricket API integration with mock match schedules
- WhatsApp API integration with test phone numbers
- CSV import/export round-trip with Hindi SKU names
- PostGIS geospatial queries for regional prior lookup

### Performance Testing

**Forecast Generation Performance**:
- Target: Generate forecasts for 100 stores in < 10 seconds
- Target: Generate forecasts for 1000 stores in < 60 seconds (with distributed processing)
- Target: Regional prior calculation for 5km radius in < 2 seconds

**Scalability Testing**:
- Test concurrent processing of 1000 stores
- Test auto-scaling trigger at 80% load
- Test cache hit rates for regional priors (target > 80%)

**Latency Testing**:
- Target: Single store forecast in < 5 seconds
- Target: Dashboard load time < 2 seconds
- Target: WhatsApp alert delivery < 10 seconds

### Test Data Strategy

**Synthetic Data Generation**:
- Generate realistic Indian sales patterns with Diwali spikes, monsoon dips, salary cycle bumps
- Generate weather data with monsoon seasonality
- Generate event data with cricket matches, festivals, bandhs at realistic frequencies
- Generate store locations across Indian states with urban/rural distribution

**Fixture Data**:
- Maintain curated datasets for specific Indian scenarios (Diwali week, IPL season, monsoon month)
- Include edge cases: single-SKU stores, high-volatility SKUs, sparse data stores
- Include multilingual SKU names (Hindi, Tamil, Bengali)

**Property Test Generators**:
- Custom Hypothesis strategies for Transaction, Forecast, Event, Store, Location objects
- Ensure generated data respects Indian business constraints (e.g., positive quantities, valid Indian pincodes, realistic festival dates)
- Include regional diversity in generated stores (different states, urban/rural mix)

### Impact Validation Testing

**Economic Impact Verification**:
- Test impact calculations with known baseline and improved scenarios
- Verify regional aggregation sums correctly
- Verify national extrapolation scales appropriately to 12 million stores
- Test ROI calculation with various cost and benefit scenarios

**Bharat Context Validation**:
- Verify festival calendar completeness for all major Indian festivals
- Verify monsoon pattern recognition across different Indian regions
- Verify cricket match impact on relevant SKU categories
- Verify bandh handling reduces and then spikes demand appropriately
