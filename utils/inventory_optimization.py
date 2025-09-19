import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InventoryOptimizer:
    def __init__(self):
        self.default_lead_time = 7  # days
        self.default_service_level = 0.95  # 95%
        self.default_holding_cost_rate = 0.2  # 20% per year
    
    def calculate_recommendations(self, forecasts_df, current_inventory_df, service_level=95, lead_time=7, safety_stock_factor=1.5):
        """Calculate inventory recommendations based on forecasts"""
        try:
            if forecasts_df is None or forecasts_df.empty:
                logger.warning("No forecast data available")
                return pd.DataFrame()
            
            if current_inventory_df is None or current_inventory_df.empty:
                logger.warning("No current inventory data available")
                return pd.DataFrame()
            
            # Convert service level to decimal
            service_level_decimal = service_level / 100.0
            
            # Get unique products from forecasts
            products = forecasts_df['product_name'].unique()
            
            recommendations = []
            
            for product in products:
                try:
                    # Get product forecasts
                    product_forecasts = forecasts_df[forecasts_df['product_name'] == product]
                    
                    if product_forecasts.empty:
                        continue
                    
                    # Get current inventory for this product
                    current_stock = self._get_current_stock(product, current_inventory_df)
                    
                    # Calculate demand statistics
                    demand_stats = self._calculate_demand_statistics(product_forecasts)
                    
                    # Calculate safety stock
                    safety_stock = self._calculate_safety_stock(
                        demand_stats, lead_time, service_level_decimal, safety_stock_factor
                    )
                    
                    # Calculate reorder point
                    reorder_point = self._calculate_reorder_point(
                        demand_stats['mean_daily_demand'], lead_time, safety_stock
                    )
                    
                    # Calculate optimal order quantity (Economic Order Quantity)
                    eoq = self._calculate_eoq(
                        demand_stats['annual_demand'],
                        demand_stats.get('ordering_cost', 50),  # Default ordering cost
                        demand_stats.get('holding_cost', 2)     # Default holding cost
                    )
                    
                    # Determine action needed
                    action, urgency_score = self._determine_action(
                        current_stock, reorder_point, safety_stock, eoq
                    )
                    
                    # Calculate recommended order quantity
                    recommended_order_qty = self._calculate_recommended_order_quantity(
                        current_stock, reorder_point, eoq, action
                    )
                    
                    # Calculate excess inventory
                    excess_inventory = max(0, current_stock - safety_stock * 2) if action == 'reduce' else 0
                    
                    recommendation = {
                        'product_name': product,
                        'current_stock': current_stock,
                        'forecasted_demand': demand_stats['mean_daily_demand'],
                        'safety_stock': safety_stock,
                        'reorder_point': reorder_point,
                        'eoq': eoq,
                        'recommended_order_quantity': recommended_order_qty,
                        'action': action,
                        'urgency_score': urgency_score,
                        'excess_inventory': excess_inventory,
                        'service_level': service_level,
                        'lead_time': lead_time
                    }
                    
                    recommendations.append(recommendation)
                
                except Exception as e:
                    logger.error(f"Error processing product {product}: {e}")
                    continue
            
            if not recommendations:
                logger.warning("No recommendations generated")
                return pd.DataFrame()
            
            recommendations_df = pd.DataFrame(recommendations)
            
            # Sort by urgency score (descending)
            recommendations_df = recommendations_df.sort_values('urgency_score', ascending=False)
            
            logger.info(f"Generated recommendations for {len(recommendations_df)} products")
            
            return recommendations_df
        
        except Exception as e:
            logger.error(f"Error calculating recommendations: {e}")
            return pd.DataFrame()
    
    def _get_current_stock(self, product, current_inventory_df):
        """Get current stock for a product"""
        try:
            product_inventory = current_inventory_df[current_inventory_df['product_name'] == product]
            
            if product_inventory.empty:
                # Default stock level if not found
                return 50.0
            
            return float(product_inventory['current_stock'].iloc[0])
        
        except Exception as e:
            logger.error(f"Error getting current stock for {product}: {e}")
            return 50.0
    
    def _calculate_demand_statistics(self, forecasts_df):
        """Calculate demand statistics from forecasts"""
        try:
            # Get predicted quantities
            predicted_quantities = forecasts_df['predicted_quantity'].values
            
            # Calculate statistics
            mean_daily_demand = np.mean(predicted_quantities)
            std_daily_demand = np.std(predicted_quantities)
            max_daily_demand = np.max(predicted_quantities)
            min_daily_demand = np.min(predicted_quantities)
            
            # Estimate annual demand
            annual_demand = mean_daily_demand * 365
            
            # Calculate coefficient of variation
            cv = std_daily_demand / mean_daily_demand if mean_daily_demand > 0 else 0
            
            stats_dict = {
                'mean_daily_demand': mean_daily_demand,
                'std_daily_demand': std_daily_demand,
                'max_daily_demand': max_daily_demand,
                'min_daily_demand': min_daily_demand,
                'annual_demand': annual_demand,
                'coefficient_of_variation': cv,
                'ordering_cost': 50,  # Default ordering cost
                'holding_cost': mean_daily_demand * 0.1  # Estimated holding cost
            }
            
            return stats_dict
        
        except Exception as e:
            logger.error(f"Error calculating demand statistics: {e}")
            return {
                'mean_daily_demand': 10,
                'std_daily_demand': 2,
                'max_daily_demand': 15,
                'min_daily_demand': 5,
                'annual_demand': 3650,
                'coefficient_of_variation': 0.2,
                'ordering_cost': 50,
                'holding_cost': 1
            }
    
    def _calculate_safety_stock(self, demand_stats, lead_time, service_level, safety_stock_factor):
        """Calculate safety stock using statistical methods"""
        try:
            # Z-score for service level
            z_score = stats.norm.ppf(service_level)
            
            # Standard deviation of demand during lead time
            std_lead_time_demand = demand_stats['std_daily_demand'] * np.sqrt(lead_time)
            
            # Basic safety stock calculation
            safety_stock = z_score * std_lead_time_demand
            
            # Apply safety stock factor
            safety_stock *= safety_stock_factor
            
            # Ensure minimum safety stock
            min_safety_stock = demand_stats['mean_daily_demand'] * 0.5  # At least half day's demand
            safety_stock = max(safety_stock, min_safety_stock)
            
            return round(safety_stock, 2)
        
        except Exception as e:
            logger.error(f"Error calculating safety stock: {e}")
            return demand_stats['mean_daily_demand'] * safety_stock_factor
    
    def _calculate_reorder_point(self, mean_daily_demand, lead_time, safety_stock):
        """Calculate reorder point"""
        try:
            # Reorder point = (Average daily demand × Lead time) + Safety stock
            reorder_point = (mean_daily_demand * lead_time) + safety_stock
            return round(reorder_point, 2)
        
        except Exception as e:
            logger.error(f"Error calculating reorder point: {e}")
            return mean_daily_demand * lead_time * 1.5
    
    def _calculate_eoq(self, annual_demand, ordering_cost, holding_cost):
        """Calculate Economic Order Quantity"""
        try:
            if holding_cost <= 0:
                holding_cost = 1  # Prevent division by zero
            
            # EOQ = sqrt((2 × Annual Demand × Ordering Cost) / Holding Cost)
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            
            return round(eoq, 2)
        
        except Exception as e:
            logger.error(f"Error calculating EOQ: {e}")
            return annual_demand / 12  # Default to monthly demand
    
    def _determine_action(self, current_stock, reorder_point, safety_stock, eoq):
        """Determine the action needed for inventory management"""
        try:
            # Calculate urgency score (0-100, higher = more urgent)
            urgency_score = 0
            action = 'maintain'
            
            if current_stock <= reorder_point:
                # Need to reorder
                action = 'reorder'
                
                # Calculate urgency based on how close we are to stockout
                if current_stock <= safety_stock:
                    urgency_score = 100  # Critical - below safety stock
                elif current_stock <= reorder_point * 0.5:
                    urgency_score = 80   # High urgency
                else:
                    urgency_score = 60   # Medium urgency
            
            elif current_stock > safety_stock * 3:
                # Potential overstock
                action = 'reduce'
                excess_ratio = current_stock / (safety_stock * 3)
                urgency_score = min(40, excess_ratio * 10)  # Low to medium urgency
            
            else:
                # Maintain current levels
                action = 'maintain'
                urgency_score = 20  # Low urgency
            
            return action, round(urgency_score, 2)
        
        except Exception as e:
            logger.error(f"Error determining action: {e}")
            return 'maintain', 20
    
    def _calculate_recommended_order_quantity(self, current_stock, reorder_point, eoq, action):
        """Calculate recommended order quantity"""
        try:
            if action == 'reorder':
                # Order quantity to reach optimal level
                optimal_stock = reorder_point + eoq
                recommended_qty = max(0, optimal_stock - current_stock)
                
                # Ensure minimum order quantity
                min_order_qty = eoq * 0.1  # At least 10% of EOQ
                recommended_qty = max(recommended_qty, min_order_qty)
                
                return round(recommended_qty, 2)
            
            elif action == 'reduce':
                # Negative quantity indicates reduction needed
                return 0
            
            else:
                # No action needed
                return 0
        
        except Exception as e:
            logger.error(f"Error calculating recommended order quantity: {e}")
            return 0
    
    def optimize_inventory_levels(self, recommendations_df, budget_constraint=None):
        """Optimize inventory levels given budget constraints"""
        try:
            if recommendations_df.empty:
                return recommendations_df
            
            optimized_df = recommendations_df.copy()
            
            if budget_constraint:
                # Sort by urgency score and fit within budget
                reorder_items = optimized_df[optimized_df['action'] == 'reorder'].copy()
                reorder_items = reorder_items.sort_values('urgency_score', ascending=False)
                
                cumulative_cost = 0
                for idx, row in reorder_items.iterrows():
                    item_cost = row['recommended_order_quantity'] * 10  # Estimated unit cost
                    
                    if cumulative_cost + item_cost <= budget_constraint:
                        cumulative_cost += item_cost
                    else:
                        # Reduce order quantity to fit budget
                        remaining_budget = budget_constraint - cumulative_cost
                        max_qty = remaining_budget / 10  # Estimated unit cost
                        optimized_df.loc[idx, 'recommended_order_quantity'] = max(0, max_qty)
                        break
            
            return optimized_df
        
        except Exception as e:
            logger.error(f"Error optimizing inventory levels: {e}")
            return recommendations_df
    
    def calculate_inventory_metrics(self, recommendations_df):
        """Calculate key inventory performance metrics"""
        try:
            if recommendations_df.empty:
                return {}
            
            metrics = {
                'total_products': len(recommendations_df),
                'reorder_needed': len(recommendations_df[recommendations_df['action'] == 'reorder']),
                'overstock_items': len(recommendations_df[recommendations_df['action'] == 'reduce']),
                'avg_service_level': recommendations_df['service_level'].mean(),
                'total_safety_stock': recommendations_df['safety_stock'].sum(),
                'total_current_stock': recommendations_df['current_stock'].sum(),
                'avg_urgency_score': recommendations_df['urgency_score'].mean(),
                'high_urgency_items': len(recommendations_df[recommendations_df['urgency_score'] >= 80])
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating inventory metrics: {e}")
            return {}
