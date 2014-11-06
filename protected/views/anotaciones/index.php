<?php
$this->breadcrumbs = array(
	'Documentos en Línea',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label' => 'Crear Doc. Online', 'url' => array('/Documentos/createOnline')),
	array('label'=>Yii::t('app', 'Gestionar Documentos') . ' Anotaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Anotaciones'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
