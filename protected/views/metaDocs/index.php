<?php
$this->breadcrumbs = array(
	'Documentos',
	Yii::t('app', 'Index'),
);

$this->menu=array(

	array('label'=>Yii::t('app', 'Gestionar Docs.') . ' Documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Documentos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
