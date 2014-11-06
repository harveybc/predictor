<?php
class JSTree extends CInputWidget{

	/**
	 * Plugins to load for the tree
	 * @var array
	 */
	public $plugins = array("themes", "html_data", "sort", "ui");

	public $bind = array();

	/**
	 * Form the array that will be fed directly into the JQuery plugin
	 *
	 * @return The array that contains the configuration of the widget
	 */
	public function makeOptions(){

		$plugins_array = array(); // We need to split out the listed plugins from their config
		$config_array = array();

		foreach($this->plugins as $plugin=>$config){ // Scroll through the array given to us by the user

			$plugins_array[] = is_numeric($plugin) ? $config : $plugin; // If the array key is numeric then the user has put no config to the plugin

			if(!is_numeric($plugin)){ // Then add this plugin to the config list
				$config_array[$plugin] = $config;
			}
		}

		return array_merge(
			$config_array, array("plugins"=>$plugins_array) // Mege the two so we have loaded plugins with their config
		);
	}

	/**
	 * @see framework/CWidget::run()
	 * @return $html The HTML of the tree object
	 */
	public function run(){
// adicionado por Harvey para poder usarse sin modelo
                if (!isset($this->attribute))
                {
                   $this->attribute="jsTreeDefaultAttribute";
                }
		list($name, $id) = $this->resolveNameID(); // Lets get the model attribute so we can make the form up

		// Lets publish the assets and get the ClientScript object so we add js etc
		$dir = dirname(__FILE__).DIRECTORY_SEPARATOR.'assets';
		$assets = Yii::app()->getAssetManager()->publish($dir);
		$cs = Yii::app()->getClientScript();

		$js_binds = '';
		foreach($this->bind as $event => $function){
			$js_binds .= CJavaScript::encode('js:$(".js_tree_'.$this->attribute.' div").bind("'.$event.'",'.$function.' );');
		}
// agregado por Harvey:
                $basePath = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'assets' . DIRECTORY_SEPARATOR;
		$baseUrl = Yii::app()->getAssetManager()->publish($basePath, false, 1, YII_DEBUG);
		$scriptFile = YII_DEBUG ? '/jquery.jstree.js' : '/jquery.jstree.js';
		$cs = Yii::app()->clientScript;
		$cs->registerCoreScript('jquery');
		$cs->registerScriptFile($baseUrl . $scriptFile);
// Fin Adición
//// Antes estaba:		$cs->registerScriptFile($assets.'/jquery.jstree.js');

		//$cs->registerScriptFile($assets.'/jquery.jstree.js');
// adicionado por harvey para cargar hotkeys
                $cs->registerScriptFile($baseUrl.'/_lib/jquery.hotkeys.js');

// fin adición
                
		$cs->registerScript('Yii.'.get_class($this).'#'.$id, '
			$(function(){
                                
                                $(".js_tree_'.$this->attribute.' div").bind("loaded.jstree", function (event, data) {
					$c_selected = [];
                                        
/* comentado por Harvey                                        
					data.inst.get_checked().each(function(i, node){
						$c_selected[$c_selected.length] = $(node).find(":checkbox").val();
					});
*/                                        
					$(this).parent().children("input").val(JSON.stringify($c_selected));
				}).jstree(
					'.CJavaScript::encode($this->makeOptions()).'
				);

				'.$js_binds.'
                                    
                                    
			});
                        
                        
                ', CClientScript::POS_READY); // Add the initial load of the JS widget to the page

		//$cs->registerScript('Yii.'.get_class($this).'#'.$id.'.binds', $js_binds);

		$html = CHTML::openTag("div", array("id"=>"miTree", "class"=>"js_tree_".$this->attribute)); // Start building the html
			
                       $html .= CHTML::openTag("div", array("id"=>"miTree2"));
			$html .= CHTML::closeTag("div");
               //adicionado por Harvey para poder usarse sin modelo ni nombre
                        if (isset($this->model))
                        {
                            $html .= CHTML::activeTextField($this->model, $this->attribute, array("style"=>"display:none;"));
                        }
                        elseif(isset($this->name))
                        {
                            $html .= CHTML::textField($this->name, "", array("style"=>"display:none;"));
                        }
                        else
                        {
                            $html .= CHTML::textField("jsTreeDefaultName", "", array("style"=>"display:none;"));
                        }
               //fin adición                        
			//$html .= CHTML::activeTextField($this->model, $this->attribute, array("style"=>"display:none;"));
		$html .= CHTML::closeTag("div");

		echo $html; // Return the full tree and all its components
	}
}