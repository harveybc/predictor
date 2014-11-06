<?php
class HighstockWidget extends CWidget {

	public $options=array();
	public $htmlOptions=array();

	/**
	 * Renders the widget.
	 */
	public function run() {
		$id = $this->getId();
		$this->htmlOptions['id'] = $id;

		echo CHtml::openTag('div', $this->htmlOptions);
		echo CHtml::closeTag('div');

		// check if options parameter is a json string
		if(is_string($this->options)) {
			if(!$this->options = CJSON::decode($this->options))
				throw new CException('The options parameter is not valid JSON.');
			// TODO translate exception message
		}

		// merge options with default values
		$defaultOptions = array('chart' => array('renderTo' => $id), 'exporting' => array('enabled' => true));
		$this->options = CMap::mergeArray($defaultOptions, $this->options);
		$jsOptions = CJavaScript::encode($this->options);
		$this->registerScripts(__CLASS__ . '#' . $id, "window.chart = new Highcharts.StockChart($jsOptions);");
	}

	/**
	 * Publishes and registers the necessary script files.
	 *
	 * @param string the id of the script to be inserted into the page
	 * @param string the embedded script to be inserted into the page
	 */
	protected function registerScripts($id, $embeddedScript) {
		$basePath = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'assets' . DIRECTORY_SEPARATOR;
		$baseUrl = Yii::app()->getAssetManager()->publish($basePath, false, 1, YII_DEBUG);
		$scriptFile = YII_DEBUG ? '/highstock.src.js' : '/highstock.js';

		$cs = Yii::app()->clientScript;
		$cs->registerCoreScript('jquery');
		$cs->registerScriptFile($baseUrl . $scriptFile);

		// register exporting module if enabled via the 'exporting' option
		if($this->options['exporting']['enabled']) {
			$scriptFile = YII_DEBUG ? 'exporting.src.js' : 'exporting.js';
			$cs->registerScriptFile("$baseUrl/modules/$scriptFile");
		}
		
		// register global theme if specified vie the 'theme' option
		if($this->options['theme']) {
			$scriptFile = $this->options['theme'] . ".js";
			$cs->registerScriptFile("$baseUrl/themes/$scriptFile");
		}
		$cs->registerScript($id, $embeddedScript, CClientScript::POS_READY);
	}
}