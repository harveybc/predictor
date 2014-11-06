<?php
$this->breadcrumbs = array(
	'documentos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Nuevo') . ' Metadocumento', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('documentos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
