<?php
$this->breadcrumbs = array(
	'Competencias Documentos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' CompetenciasDocumentos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' CompetenciasDocumentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Competencias Documentos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
